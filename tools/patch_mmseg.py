import importlib.util
import sys

def patch_mmseg():
    # Find the mmseg package without executing __init__.py (which would trigger the crash)
    spec = importlib.util.find_spec('mmseg')
    if not spec or not spec.origin:
        print("Error: Could not locate mmseg. Are you in the 'clustvit_env' conda environment?")
        sys.exit(1)

    filepath = spec.origin
    print(f"Targeting: {filepath}")

    with open(filepath, 'r') as file:
        content = file.read()

    # The exact string OpenMMLab uses for the version check
    target_str = "mmcv_min_version <= mmcv_version < mmcv_max_version"
    
    if "assert (True)" in content and target_str not in content:
        print("File is already patched!")
        return

    if target_str in content:
        # We replace the condition with 'True' inside the existing parentheses.
        # This prevents a SyntaxError with the multi-line backslashes (\) that follow it.
        new_content = content.replace(target_str, "True")
        
        with open(filepath, 'w') as file:
            file.write(new_content)
        print("Successfully patched! The MMCV version check is now permanently disabled.")
    else:
        print("Could not find the exact version check. It may have been updated or altered manually.")

if __name__ == "__main__":
    patch_mmseg()