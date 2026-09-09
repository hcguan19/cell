"""Synthetic-only contracts for archived numerical functions.

AST extraction deliberately avoids executing original module-level data reads
and heavyweight GPU imports. These are local unit tests, not full reproduction.
"""
from pathlib import Path
import ast
import json
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'research/cell_train3'


def functions(path, names, extra=None):
    """只执行明确指定的函数定义，不运行原文件main或顶层读写。"""
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in body} != set(names):
        raise AssertionError('requested function not found')
    env = {'np': np, 'pd': pd, 'json': json}
    env.update(extra or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), env)
    return env


class DecoderContractTests(unittest.TestCase):
    def setUp(self):
        self.env = functions('cycle_count_topk_bestvideo_v1/run_best_video_topk.py',
                             ['boundary_rank_gap', 'decode', 'predict_distributions', 'expand_ordinal'])
        # Artificial IDs and values only; non-contiguous index checks position handling.
        self.frame = pd.DataFrame({'group_id': ['synthetic'] * 3,
                                   'p_classify': [.4, .4, .4],
                                   'video_rank': [.2, .9, .1]}, index=[10, 30, 70])
        self.dist = pd.DataFrame({'group_id': ['synthetic'], 'n': [3], 'k_mode': [1],
                                  'post_max': [.8], 'post_margin': [.6], 'post_entropy': [.2]})

    def decode(self, frame=None, **kwargs):
        options = dict(post_thr=.2, margin_thr=.02, entropy_thr=.95,
                       rank_gap_thr=0., max_changes=1)
        options.update(kwargs)
        return self.env['decode'](self.frame if frame is None else frame, self.dist, **options)

    def test_top_one_and_score_threshold_agree(self):
        pred, prob, acted = self.decode()
        np.testing.assert_array_equal(pred, [0, 1, 0])
        np.testing.assert_array_equal(prob >= .5, pred)
        self.assertTrue(acted.all())

    def test_unreachable_threshold_preserves_table(self):
        pred, prob, acted = self.decode(post_thr=2.)
        np.testing.assert_array_equal(pred, [0, 0, 0])
        np.testing.assert_array_equal(prob, [.4, .4, .4])
        self.assertFalse(acted.any())

    def test_no_allowed_count_change_falls_back(self):
        pred, _, acted = self.decode(max_changes=0)
        self.assertEqual(pred.sum(), 0)
        self.assertFalse(acted.any())

    def test_extreme_count_does_not_act(self):
        self.dist['k_mode'] = 3
        _, _, acted = self.decode(max_changes=3)
        self.assertFalse(acted.any())

    def test_tied_scores_have_stable_input_order(self):
        frame = self.frame.copy()
        frame['video_rank'] = .5
        pred, _, _ = self.decode(frame)
        np.testing.assert_array_equal(pred, [1, 0, 0])

    def test_decoder_does_not_read_labels(self):
        a = self.frame.assign(label=[0, 1, 0])
        b = self.frame.assign(label=[1, 0, 1])
        for x, y in zip(self.decode(a), self.decode(b)):
            np.testing.assert_array_equal(x, y)

    def test_acted_is_not_changed_sample_count(self):
        pred, _, acted = self.decode()
        self.assertEqual(int(acted.sum()), 3)
        self.assertEqual(int((pred != 0).sum()), 1)

    def test_ordinal_targets(self):
        cycles = pd.DataFrame({'group_id': ['synthetic'], 'p': [.4], 'logit_p': [-.4],
                               'n': [3], 'np': [1.2], 'pgt_type': ['type-a'], 'k_true': [2]})
        expanded = self.env['expand_ordinal'](cycles)
        np.testing.assert_array_equal(expanded.target, [1, 1, 0])

    def test_survival_projection_is_a_distribution(self):
        class DummyModel:
            def predict_proba(self, x):
                # Deliberately non-monotone; k_true must not enter feature matrix.
                assert 'k_true' not in x.columns
                p = np.array([.7, .8, .1])
                return np.column_stack([1-p, p])
        self.env['NUM'] = ['p', 'logit_p', 'n', 'np', 'j', 'j_frac', 'p_minus_jfrac']
        self.env['CAT'] = ['pgt_type']
        cycles = pd.DataFrame({'group_id': ['synthetic'], 'fold': [0], 'p': [.4],
                               'logit_p': [-.4], 'n': [3], 'np': [1.2],
                               'pgt_type': ['type-a'], 'k_true': [2]})
        out = self.env['predict_distributions'](DummyModel(), cycles)
        posterior = json.loads(out.posterior.iloc[0])
        np.testing.assert_allclose(posterior, [.3, 0, .6, .1])
        self.assertAlmostEqual(sum(posterior), 1.)


class FoldAndRankTests(unittest.TestCase):
    def test_count_inner_fit_excludes_outer_and_validation(self):
        calls = []
        def fit(frame, c):
            return set(frame.fold)
        def predict(model, frame):
            calls.append((model, set(frame.fold)))
            return frame.copy()
        env = functions('cycle_count_topk_bestvideo_v1/run_best_video_topk.py', ['crossfit_dist'],
                        {'cycle_frame': lambda x: x, 'fit_ordinal': fit,
                         'predict_distributions': predict})
        inner, test = env['crossfit_dist'](pd.DataFrame({'fold': range(5)}), 2, .2)
        self.assertEqual(set(inner.fold), {0, 1, 3, 4})
        self.assertEqual(set(test.fold), {2})
        for trained, predicted in calls:
            self.assertNotIn(2, trained)
            self.assertFalse(trained & predicted)

    def test_pairwise_examples_are_same_group_and_symmetric(self):
        env = functions('evaluate_vjepa21_nested_rank_topk.py', ['paired_examples'])
        x, y = env['paired_examples'](np.array([[3.], [1.], [99.]]),
                                     np.array([1, 0, 1]), np.array(['a', 'a', 'b']))
        np.testing.assert_array_equal(x, [[2.], [-2.]])
        np.testing.assert_array_equal(y, [1, 0])

    def test_no_mixed_group_fails_explicitly(self):
        env = functions('evaluate_vjepa21_nested_rank_topk.py', ['paired_examples'])
        with self.assertRaises(ValueError):
            env['paired_examples'](np.ones((2, 1)), np.ones(2), np.array(['a', 'b']))


if __name__ == '__main__':
    unittest.main()
