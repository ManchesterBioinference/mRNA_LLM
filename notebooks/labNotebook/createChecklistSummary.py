import re
import argparse

def slugify(text):
    """Create a stable anchor ID."""
    text = text.strip().lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    return text

def generate_unique_anchor(text, used_anchors):
    base = slugify(text)
    anchor = base
    i = 1
    while anchor in used_anchors:
        anchor = f"{base}-{i}"
        i += 1
    used_anchors.add(anchor)
    return anchor

def main():
    parser = argparse.ArgumentParser(description="Process a Markdown file to add a checklist summary and anchors.")
    parser.add_argument("markdown_file", help="Path to the Markdown file to process.")
    args = parser.parse_args()

    markdown_file_path = args.markdown_file

    try:
        with open(markdown_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {markdown_file_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    checklist_summary = []
    used_anchors = set()
    processed_lines = [] # Use a new list for lines with potential anchor insertions
    anchor_map = {} # To store original line index to anchor mapping

    # First pass: Identify existing anchors and checklist items, generate new anchors
    temp_lines_for_anchor_gen = list(lines) # Temporary list to simulate insertions for anchor generation

    i = 0
    original_line_idx = 0
    while original_line_idx < len(lines):
        line = lines[original_line_idx]
        match = re.match(r"^\s*[*-] \[([ ])\] (.+)", line)
        if match:
            status = match.group(1)
            text = match.group(2).strip()
            anchor = None

            # Look backward for an existing anchor or heading in original lines
            # Check current line first for an existing anchor
            current_line_anchor_match = re.search(r'<a name="([^"]+)">', line)
            if current_line_anchor_match:
                anchor = current_line_anchor_match.group(1)
                used_anchors.add(anchor)
            else:
                # Look in preceding lines
                for j in range(original_line_idx - 1, max(original_line_idx - 6, -1), -1):
                    # Check if line j has an anchor already
                    # This needs to check the potentially modified lines if we were inserting anchors on the fly
                    # For simplicity in this pass, we check original lines, but this might need refinement
                    # if anchors are inserted *before* the item they refer to.
                    # Let's assume anchors are on the same line or the line immediately before the heading.

                    # Check for anchor on the line itself (e.g. <a name="foo"></a> - [x] item)
                    # This is less common for checklist items but good to check.
                    # More common is anchor before heading, or heading itself.

                    # Check for an anchor tag on the line before the checklist item
                    if j >= 0: # Ensure j is a valid index
                        prev_line_anchor_match = re.search(r'<a name="([^"]+)">', lines[j])
                        if prev_line_anchor_match:
                            anchor = prev_line_anchor_match.group(1)
                            used_anchors.add(anchor)
                            break

                    # Check for a heading to slugify
                    heading_match = re.match(r"^(#+) (.+)", lines[j] if j >= 0 else "")
                    if heading_match:
                        anchor = slugify(heading_match.group(2))
                        # We don't add to used_anchors here yet, as it might be a new one
                        # generate_unique_anchor will handle it.
                        break


            if not anchor:
                # If no anchor found, generate one.
                # We need to ensure the anchor is unique *before* inserting it.
                # The anchor should be associated with the checklist item.
                # The anchor text itself will be inserted *before* the checklist item in the `processed_lines`.
                anchor = generate_unique_anchor(text, used_anchors)
                # We will insert the anchor tag in the second pass

            checklist_summary.append(f'- [{"x" if status.lower() == "x" else " "}] [{text}](#{anchor})')
            anchor_map[original_line_idx] = anchor # Map original line index to its anchor

        original_line_idx += 1


    # Second pass: Reconstruct the file content with new anchors and the summary
    output_lines = []
    summary_inserted = False
    first_h1_found = False
    summary_block_str = (
        "<details>\n<summary><B>Checklist Summary<B></summary>\n\n"
        + "\n".join(checklist_summary)
        + "\n\n</details>\n\n"
    )

    # Remove existing summary block if present
    temp_output_lines = []
    in_old_summary = False
    for line_idx, line_content in enumerate(lines):
        if "<details>" in line_content and "Checklist Summary" in line_content:
            in_old_summary = True
            continue
        if in_old_summary and "</details>" in line_content:
            in_old_summary = False
            continue
        if not in_old_summary:
            temp_output_lines.append(line_content)
    lines = temp_output_lines


    for original_line_idx, line_content in enumerate(lines):
        # Insert summary after the first H1
        if not summary_inserted and re.match(r"^#\s", line_content):
            if not first_h1_found:
                output_lines.append(line_content) # Add the H1 line itself
                if checklist_summary: # Only add summary if there are items
                    output_lines.extend(summary_block_str.splitlines(keepends=True))
                summary_inserted = True
                first_h1_found = True
                continue # Continue to next line after inserting summary

        # Add new anchor if this line was a checklist item and didn't have one
        if original_line_idx in anchor_map:
            # Check if an anchor already exists on this line or the one before (for headings)
            # This logic is to avoid duplicate anchors if one was already present.
            # A more robust way would be to check if the *specific* generated anchor needs to be added.
            anchor_to_check = anchor_map[original_line_idx]
            line_has_anchor = f'name="{anchor_to_check}"' in line_content
            prev_line_has_anchor = False
            if output_lines: # Check previous line in output_lines
                prev_line_has_anchor = f'name="{anchor_to_check}"' in output_lines[-1]

            is_heading_associated_with_anchor = False
            if original_line_idx > 0:
                heading_match_prev = re.match(r"^(#+) (.+)", lines[original_line_idx-1])
                if heading_match_prev and slugify(heading_match_prev.group(2)) == anchor_to_check:
                    is_heading_associated_with_anchor = True


            # Only add the anchor if it's not already present on the line,
            # or on the preceding line (if it's a heading that got slugified to this anchor)
            # and it's not a slugified heading that implicitly serves as the anchor.
            # The condition for adding an anchor is:
            # 1. The line is a checklist item (original_line_idx in anchor_map).
            # 2. An anchor for this item was generated (anchor_map[original_line_idx] exists).
            # 3. The line itself doesn't already contain this specific anchor.
            # 4. The *previous* line (if it was a heading that generated this anchor) doesn't contain it.
            #    (This is tricky, slugify might match an existing manual anchor).
            #    A simpler rule: if we generated an anchor for an item, and that item
            #    didn't have an explicit <a name="..."> tag immediately preceding it or on its line,
            #    and it wasn't immediately preceded by a heading that slugifies to that anchor, then insert.

            # Let's refine: Insert anchor if it was generated for a checklist item *and*
            # no anchor tag <a name="generated_anchor_name"> exists immediately before or on the item's line.
            # And no heading that slugifies to this anchor exists immediately before.

            needs_explicit_anchor_tag = True
            # Check if checklist item line itself has the anchor
            if f'name="{anchor_to_check}"' in line_content:
                needs_explicit_anchor_tag = False

            # Check if line immediately before checklist item has the anchor tag
            # (This applies if the anchor was inserted for a non-heading item)
            if output_lines and f'<a name="{anchor_to_check}">' in output_lines[-1]:
                 needs_explicit_anchor_tag = False

            # Check if line immediately before is a heading that slugifies to this anchor
            # (This means the heading itself acts as the anchor, no need for <a name...>)
            if original_line_idx > 0:
                prev_line_content = lines[original_line_idx-1]
                heading_match_on_prev = re.match(r"^(#+) (.+)", prev_line_content)
                if heading_match_on_prev:
                    if slugify(heading_match_on_prev.group(2).strip()) == anchor_to_check:
                        needs_explicit_anchor_tag = False


            if needs_explicit_anchor_tag:
                 # Check if the anchor was generated because no suitable heading/anchor was found *before* the item
                is_newly_generated_for_item = False
                # Heuristic: if slugify(item_text) == anchor, it was likely generated for the item itself
                # This needs to be more robust by tracking *why* an anchor was generated.
                # For now, if an anchor is in anchor_map, and it's not from a preceding heading, assume it needs insertion.

                # Simplified: if an anchor is in anchor_map, and it's not already on the line or the line before (as a tag),
                # and it's not a slugified heading right before, then add it.
                # This part is tricky because `generate_unique_anchor` is called if *no* anchor was found.
                # So, if `anchor_map[original_line_idx]` exists, it means an anchor was determined for this item.
                # We need to ensure we only add the `<a name...>` tag if it's *not* from a slugified heading.

                # Re-evaluate if anchor came from a heading
                anchor_from_heading = False
                for j in range(original_line_idx - 1, max(original_line_idx - 6, -1), -1):
                    if j < 0: continue
                    heading_match_lookback = re.match(r"^(#+) (.+)", lines[j])
                    if heading_match_lookback:
                        if slugify(heading_match_lookback.group(2)) == anchor_to_check:
                            anchor_from_heading = True
                        break # Found nearest heading
                    if re.search(r'<a name="([^"]+)">', lines[j]): # Stop if another anchor is found
                        break


                if not anchor_from_heading and needs_explicit_anchor_tag:
                    output_lines.append(f'<a name="{anchor_to_check}"></a>\n')

        output_lines.append(line_content)

    # If no H1 was found, prepend the summary (if any checklist items)
    if not first_h1_found and checklist_summary:
        output_lines = summary_block_str.splitlines(keepends=True) + output_lines


    try:
        with open(markdown_file_path, "w", encoding="utf-8") as f:
            f.writelines(output_lines)
        print(f"✔️ Checklist summary updated and anchors processed in {markdown_file_path}.")
    except Exception as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    main()
