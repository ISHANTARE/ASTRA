import os
import re

def clean_comments_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    
    # Regex to match tags like [FM-9 Fix], [T-01 fix], [FIX], [P-01 Fix], etc.
    # We will just strip the tag if the comment continues, or remove the comment
    # if it's purely about the fix.
    tag_pattern = re.compile(r'#\s*\[[A-Za-z0-9\-\s/—]+(?:[Ff]ix|follow-up|Audit)[^\]]*\]\s*')
    # Also generic things like [A-03]
    generic_tag_pattern = re.compile(r'#\s*\[[A-Z0-9\-]+\]\s*(?:[Ff]ix.*?:?)?\s*')
    
    # Another pattern for "Finding #X" or "Fix:"
    finding_pattern = re.compile(r'(?i)#\s*(?:FM|T|LOW|MED|HIGH|D|F|C|A|P)-\d+\s*fix\b')
    
    for line in lines:
        original = line
        
        # If line contains a comment
        if '#' in line:
            # We want to be careful not to match inside strings, but for simplicity, 
            # we'll assume # starts a comment in most cases (safe in this codebase unless it's a specific string).
            parts = line.split('#', 1)
            before_comment = parts[0]
            comment = '#' + parts[1]
            
            # Clean the comment
            cleaned_comment = tag_pattern.sub('# ', comment)
            cleaned_comment = generic_tag_pattern.sub('# ', cleaned_comment)
            
            # Remove "finding #..."
            cleaned_comment = re.sub(r'(?i)finding #\d+(/\d+)?\s*-?\s*', '', cleaned_comment)
            cleaned_comment = re.sub(r'(?i)finding #\d+(/\d+)?\s*audit\s*-?\s*', '', cleaned_comment)
            cleaned_comment = re.sub(r'(?i)use\s+module-level constants imported from.*', '', cleaned_comment)
            
            cleaned_comment = cleaned_comment.replace('# - ', '# ')
            cleaned_comment = cleaned_comment.replace('# — ', '# ')
            cleaned_comment = cleaned_comment.replace('# :', '# ')
            
            # If comment becomes empty or just " ", remove it
            if cleaned_comment.strip() in ('#', '# '):
                line = before_comment.rstrip()
                if line or not before_comment.strip(): # keep line if it was empty, or if code remains
                    line = line + '\n'
            else:
                line = before_comment + cleaned_comment
                
            # If the comment is literally just "fix: ... " or "follow-up: ...", maybe strip it too
            # Let's just rely on the tag removal first to see how much it cleans.
            
            # Additional cleanup for specific garbage left over
            if re.match(r'^\s*#\s*$', line):
                continue
                
        new_lines.append(line)

    # Re-evaluate to drop lines that are just `#` after processing
    final_lines = []
    for line in new_lines:
        if line.strip() == '#' or line.strip() == '':
            if original.strip() == '':
                final_lines.append(line)
        else:
            final_lines.append(line)

    if new_lines != lines:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(final_lines)
        print(f"Cleaned {filepath}")

def main():
    astra_dir = 'astra'
    for root, _, files in os.walk(astra_dir):
        for file in files:
            if file.endswith('.py'):
                clean_comments_in_file(os.path.join(root, file))
                
    tests_dir = 'tests'
    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith('.py'):
                clean_comments_in_file(os.path.join(root, file))

if __name__ == '__main__':
    main()
