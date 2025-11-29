with open('DRM_Project.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Check Cell 34 (BSM Greeks)
cell34_start = content.find('TASK B: BSM MODEL')
cell34_section = content[cell34_start:cell34_start+1500]
has_check_34 = 'VOLATILITY SAFETY CHECK' in cell34_section

# Check Cell 35 (Binomial)
cell35_start = content.find('TASK C: BINOMIAL')
cell35_section = content[cell35_start:cell35_start+1500]
has_check_35 = 'VOLATILITY SAFETY CHECK' in cell35_section

print(f"Cell 34 (BSM Greeks) has safety check: {has_check_34}")
print(f"Cell 35 (Binomial) has safety check: {has_check_35}")

if has_check_34 and has_check_35:
    print("\n✓ Both cells now have volatility safety checks!")
    print("  They will auto-correct 1375% → 25.57% when executed")
else:
    print("\n❌ Safety checks missing!")
