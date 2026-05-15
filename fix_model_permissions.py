import os
import shutil
import stat

# Try to change permissions and copy
model_dir = "model_working_copy.keras"
model_weights = os.path.join(model_dir, "model.weights.h5")

print(f"Checking {model_dir}...")
print(f"Contents: {os.listdir(model_dir)}")
print(f"Weights file exists: {os.path.exists(model_weights)}")

# Try to change permissions
try:
    for root, dirs, files in os.walk(model_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.chmod(dir_path, stat.S_IRWXU)
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.chmod(file_path, stat.S_IRWXU)
    print("✅ Permissions changed")
except Exception as e:
    print(f"⚠️ Permission change failed: {e}")

# Try to copy to a more accessible location
try:
    if os.path.exists(model_weights):
        shutil.copy(model_weights, "updated_model.h5")
        print("✅ Copied weights to updated_model.h5")
except Exception as e:
    print(f"⚠️ Copy failed: {e}")
