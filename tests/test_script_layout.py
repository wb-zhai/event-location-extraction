import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('event_run', ROOT / 'scripts/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ScriptLayoutTests(unittest.TestCase):
    def test_all_aliases_resolve_inside_repository(self):
        paths = json.loads((ROOT / 'configs/script_paths.json').read_text())
        for old, new in paths.items():
            with self.subTest(old=old):
                self.assertEqual(runner.resolve_script(old), ROOT / new)

    def test_queue_defaults_use_repository_outputs(self):
        paths = json.loads((ROOT / 'configs/script_paths.json').read_text())
        location = ROOT / paths['build_location_layer_queue.py']
        spec = importlib.util.spec_from_file_location('queue_defaults', location)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.REPO, ROOT)
        self.assertEqual(module.DB, ROOT / 'outputs/geography/global_v4_audit/location_layer_pairs.sqlite')

    def test_postprocessor_defaults_find_relocated_inputs(self):
        paths = json.loads((ROOT / 'configs/script_paths.json').read_text())
        tree = ast.parse((ROOT / paths['global_v4_postprocess.py']).read_text())
        flags = {'--parser', '--audit-report', '--shard-stats', '--missing-ids', '--exclusions'}
        checked = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument':
                continue
            names = {a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
            for flag in names & flags:
                default = next(k.value.value for k in node.keywords if k.arg == 'default')
                self.assertTrue((ROOT / default).is_file(), default)
                checked.add(flag)
        self.assertEqual(checked, flags)
    def test_root_contains_no_python_entry_points(self):
        self.assertEqual(list(ROOT.glob('*.py')), [])

    def test_reject_outside_repository(self):
        with self.assertRaises(ValueError):
            runner.resolve_script('../llm-risk-factors-extraction/scripts/run.py')

    def test_postprocessing_help_from_another_directory(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/run.py'),
                                 'global_v4_postprocess.py', '--help'],
                                cwd=ROOT.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--audit-report', result.stdout)

    def test_historical_inference_helpers_exist_in_current_module(self):
        helper = ROOT / 'scripts/event_extraction/generation/to_sft.py'
        names = {node.name for node in ast.parse(helper.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        paths = json.loads((ROOT / 'configs/script_paths.json').read_text())
        for old in ('v1_vllm_infer_20260617.py', 'v1_production_21055/vllm_infer.py'):
            for node in ast.walk(ast.parse((ROOT / paths[old]).read_text())):
                if isinstance(node, ast.ImportFrom) and node.module == 'scripts.event_extraction.generation.to_sft':
                    self.assertTrue({name.name for name in node.names} <= names)

    def test_historical_docker_copy_inputs_exist(self):
        base = ROOT / 'scripts/event_extraction/inference/historical/v1_production_21055'
        for line in (base / 'Dockerfile').read_text().splitlines():
            if line.startswith('COPY '):
                self.assertTrue((ROOT / line.split()[1]).exists(), line)
