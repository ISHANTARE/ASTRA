import os
import re

def clean_comments_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove [XXX-01 fix] style tags completely
    # E.g. [CRIT-01 / HIGH-03 fix], [MED-05 fix], [FM-1/FM-6 Fix — /#21]
    content = re.sub(r'(?i)\[[A-Z0-9\-\s/—]+(?:Fix|fix|Audit)[^\]]*\]\s*', '', content)

    # 2. Remove AUDIT-B-01 Fix: and similar
    content = re.sub(r'(?i)AUDIT-[A-Z0-9\-]+\s*Fix:\s*', '', content)

    # 3. Process line by line to handle "Previously..." sentences
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if 'Previously' in line or 'previously' in line or 'parity with' in line:
            # If the entire line is a comment or docstring line about "Previously"
            line_stripped = line.strip()
            if line_stripped.startswith('#') or line_stripped.startswith('"""') or line_stripped.startswith("'''"):
                # if it just says "# Previously...", skip it
                if line_stripped.lower().startswith('# previously') or line_stripped.lower().startswith('#  previously'):
                    continue
        
        # Another specific cleanup for trailing garbage from earlier replacements
        line = re.sub(r'\s*#\s*$', '', line)
        line = re.sub(r'\s*#\s*—\s*$', '', line)
        line = re.sub(r'\s*#\s*—\s*\]', '', line)
        
        if line.strip() == '#' or line.strip() == '# ':
            continue

        new_lines.append(line)

    final_content = '\n'.join(new_lines)
    
    # Clean up multi-line docstring sentences starting with "Previously"
    # This regex looks for "Previously... ." and removes the sentence.
    final_content = re.sub(r'(?im)^[\s#]*Previously[^.]*\.\s*', '', final_content)

    if content != final_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(final_content)
        print(f"Cleaned {filepath}")

def main():
    for root, _, files in os.walk('astra'):
        for file in files:
            if file.endswith('.py'):
                clean_comments_in_file(os.path.join(root, file))
                
    for root, _, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                clean_comments_in_file(os.path.join(root, file))

if __name__ == '__main__':
    main()
