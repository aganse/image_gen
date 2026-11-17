"""
Tests of the params/params_<name>.json files.  These need no ML packages, so
they run anywhere: each file must be valid JSON with the three expected
groups, have a matching train_<name>.py script, use only parameter names that
script defines, and fit SageMaker's hyperparameter size limit once encoded
the way `make sm-submit` encodes it.
"""

import ast
import base64
import glob
import json
import os
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = ("run_params", "training_params", "model_arch_params")
SAGEMAKER_HYPERPARAM_MAX_CHARS = 2500


def _param_keys_in_script(script_path):
    """Collect every string key of dict literals / .update({..}) in a script."""
    with open(script_path) as f:
        tree = ast.parse(f.read())
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


class TestParamsFiles(unittest.TestCase):
    """Verify consistency of params/*.json with the train_*.py scripts."""

    @classmethod
    def setUpClass(cls):
        cls.params_files = sorted(
            glob.glob(os.path.join(REPO_ROOT, "params", "params_*.json"))
        )

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def _name(self, path):
        return os.path.basename(path)[len("params_"):-len(".json")]

    def test_params_files_exist(self):
        names = {self._name(p) for p in self.params_files}
        self.assertEqual(names, {"vae", "dcgan", "wgangp", "diffusion"})

    def test_valid_json_with_three_groups(self):
        for path in self.params_files:
            with open(path) as f:
                params = json.load(f)
            self.assertEqual(set(params), set(GROUPS), msg=path)

    def test_matching_script_exists(self):
        for path in self.params_files:
            script = os.path.join(REPO_ROOT, f"train_{self._name(path)}.py")
            self.assertTrue(os.path.exists(script), msg=script)

    def test_script_names_are_alphanumeric(self):
        # SCRIPT names become part of SageMaker job names (no underscores).
        for path in self.params_files:
            self.assertTrue(self._name(path).isalnum(), msg=path)

    def test_keys_are_known_to_script(self):
        for path in self.params_files:
            name = self._name(path)
            known = _param_keys_in_script(os.path.join(REPO_ROOT, f"train_{name}.py"))
            if name == "wgangp":  # builds on the DCGAN defaults
                known |= _param_keys_in_script(os.path.join(REPO_ROOT, "train_dcgan.py"))
            with open(path) as f:
                params = json.load(f)
            for group in GROUPS:
                unknown = set(params[group]) - known
                self.assertFalse(unknown, msg=f"{path} [{group}]: {unknown}")

    def test_no_key_appears_in_two_groups(self):
        # The scripts merge the three groups into one kwargs set for printing.
        for path in self.params_files:
            with open(path) as f:
                params = json.load(f)
            keys = [key for group in GROUPS for key in params[group]]
            duplicates = {key for key in keys if keys.count(key) > 1}
            self.assertFalse(duplicates, msg=f"{path}: {duplicates}")

    def test_environment_specific_settings_not_in_files(self):
        # images_path comes from the IMAGES_PATH env var, never the params file.
        for path in self.params_files:
            with open(path) as f:
                params = json.load(f)
            self.assertNotIn("images_path", params["run_params"], msg=path)

    def test_encoded_size_within_sagemaker_limit(self):
        for path in self.params_files:
            with open(path) as f:
                params = json.load(f)
            minified = json.dumps(params, separators=(",", ":"))
            encoded = base64.b64encode(minified.encode()).decode()
            self.assertLessEqual(
                len(encoded), SAGEMAKER_HYPERPARAM_MAX_CHARS, msg=path
            )


if __name__ == "__main__":
    unittest.main()
