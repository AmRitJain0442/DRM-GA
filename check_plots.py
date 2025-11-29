with open('DRM_Project.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Check for savefig before plt.show()
for i in [333, 700, 1618, 1917, 2149, 2685]:
    print(f"\n=== Line {i+1} ===")
    print(''.join(lines[max(0,i-2):i+3]))
