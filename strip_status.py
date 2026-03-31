def remove_lines(filepath, start, end):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 1-indexed to 0-indexed translation
    start_idx = start - 1
    end_idx = end  # End is inclusive, so we slice up to `end`
    
    del lines[start_idx:end_idx]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

# Remove checkApiStatus from files:
remove_lines('e:/AgroNexus/ui/index.html', 464, 495)
remove_lines('e:/AgroNexus/ui/demo.html', 325, 358)
remove_lines('e:/AgroNexus/ui/control.html', 599, 631)

print("Status checks removed from all 3 HTML files.")
