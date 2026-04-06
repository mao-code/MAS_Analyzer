import shutil
import subprocess
import unittest


class TestMainPython310Import(unittest.TestCase):
    def test_main_imports_under_python3_when_python3_is_pre311(self) -> None:
        python3 = shutil.which("python3")
        if not python3:
            self.skipTest("python3 is unavailable")

        version = subprocess.run(
            [python3, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True,
            text=True,
            check=True,
        )
        major_minor = tuple(int(part) for part in version.stdout.strip().split("."))
        if major_minor >= (3, 11):
            self.skipTest("python3 is not a pre-3.11 interpreter in this environment")

        result = subprocess.run(
            [python3, "-c", "import main; print(main._now_stamp())"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertRegex(result.stdout.strip(), r"^\d{8}T\d{6}Z$")


if __name__ == "__main__":
    unittest.main()
