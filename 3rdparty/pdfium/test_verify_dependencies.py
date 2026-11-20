"""Exercise the source dependency checks using a disposable Git repository."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('verify_dependencies', Path(__file__).with_name('verify-dependencies.py'))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class DependencyVerificationTest(unittest.TestCase):
    def test_toolchain_tree_content_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'tool'
            source.write_text('one')
            original = verifier.tree_digest(root)
            source.write_text('two')
            self.assertNotEqual(original, verifier.tree_digest(root))

    def test_revision_and_worktree(self):
        git = shutil.which('git')
        self.assertIsNotNone(git)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / 'dependency'
            repo.mkdir()
            def command(*args):
                return subprocess.check_output([git, '-C', str(repo), *args], text=True).strip()
            command('init', '-q')
            (repo / 'source.txt').write_text('original\n')
            (repo / '.gitignore').write_text('ignored.txt\n')
            command('add', 'source.txt', '.gitignore')
            command('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')
            revision = command('rev-parse', 'HEAD')
            verifier.verify(root, git, {'dependency': revision})
            with self.assertRaisesRegex(RuntimeError, 'revision mismatch'):
                verifier.verify(root, git, {'dependency': '0' * 40})
            (repo / 'source.txt').write_text('changed\n')
            with self.assertRaisesRegex(RuntimeError, 'tracked or untracked'):
                verifier.verify(root, git, {'dependency': revision})
            command('checkout', '--', 'source.txt')
            (repo / 'new.txt').write_text('new\n')
            with self.assertRaisesRegex(RuntimeError, 'tracked or untracked'):
                verifier.verify(root, git, {'dependency': revision})
            (repo / 'new.txt').unlink()
            (repo / 'ignored.txt').write_text('ignored input\n')
            with self.assertRaisesRegex(RuntimeError, 'tracked or untracked'):
                verifier.verify(root, git, {'dependency': revision})


if __name__ == '__main__':
    unittest.main()
