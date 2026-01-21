
lines_to_process = []
file_path = r"d:\GIT\Mike-Rowley\ComfyUI_Soze_Gemini\gemini_node.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Range is 0-indexed in list, but user lines are 1-based.
# Shift lines 779 to 1062 (inclusive) left by 4 spaces.
# 779 1-based -> index 778
# 1062 1-based -> index 1061

start_idx = 778
end_idx = 1062 # slice end is exclusive, so 1062 means up to 1061 inclusive

# Verify the lines look like what we expect before modifying
print(f"Line {start_idx+1}: {repr(lines[start_idx])}")
print(f"Line {end_idx}: {repr(lines[end_idx-1])}")

# Check if indentation is correct for dedenting
if not lines[start_idx].startswith(' ' * 16):
    print("Warning: Start line does not have 16 spaces indentation")
    
# Process
for i in range(start_idx, end_idx):
    if lines[i].strip(): # Only dedent non-empty lines
        if lines[i].startswith('    '):
            lines[i] = lines[i][4:]
        else:
            print(f"Warning: Line {i+1} starts with less than 4 spaces: {repr(lines[i])}")

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
    
print("Done")
