import subprocess
import glob
import sys

print("======================================")
print(" FULL ECONOPHYSICS PIPELINE STARTED ")
print("======================================\n")

# --------------------------------------------------
# 1️⃣ Run all country build scripts automatically
# --------------------------------------------------

build_files = glob.glob("*_build.py")

if not build_files:
    print("No country build files found.")
else:
    for file in build_files:
        print(f"\nRunning {file} ...")
        result = subprocess.run([sys.executable, file])
        
        if result.returncode != 0:
            print(f"❌ ERROR in {file}. Stopping pipeline.")
            sys.exit(1)

print("\n✅ All country builds completed.")

# --------------------------------------------------
# 2️⃣ Build Final Panel
# --------------------------------------------------

print("\nRunning build_final_panel.py ...")
result = subprocess.run([sys.executable, "build_final_panel.py"])

if result.returncode != 0:
    print("❌ ERROR in build_final_panel.py")
    sys.exit(1)

print("✅ Final panel created.")

# --------------------------------------------------
# 3️⃣ Run Final FE Model
# --------------------------------------------------

print("\nRunning panel_final_fe_locked_v2.py ...")
result = subprocess.run([sys.executable, "panel_final_fe_locked_v2.py"])

if result.returncode != 0:
    print("❌ ERROR in FE model.")
    sys.exit(1)

print("\n======================================")
print(" PIPELINE COMPLETED SUCCESSFULLY ")
print("======================================")