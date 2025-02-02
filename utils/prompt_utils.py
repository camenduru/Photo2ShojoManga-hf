def remove_duplicates(base_prompt):
    # タグの重複を取り除く
    prompt_list = base_prompt.split(", ")
    seen = set()
    unique_tags = []
    for tag in prompt_list :
        tag_clean = tag.lower().strip()
        if tag_clean not in seen and tag_clean != "":
            unique_tags.append(tag)
            seen.add(tag_clean)
    return ", ".join(unique_tags)


def remove_color(base_prompt):
    # タグの色情報を取り除く
    prompt_list = base_prompt.split(", ")
    color_list = ["pink", "red", "orange", "brown", "yellow", "green", "blue", "purple", "blonde", "colored skin", "white hair", "grey", "black"]
    # カラータグを除去します。
    cleaned_tags = [tag for tag in prompt_list if all(color.lower() not in tag.lower() for color in color_list)]
    return ", ".join(cleaned_tags)


def execute_prompt(execute_tags, base_prompt):
    prompt_list = base_prompt.split(", ")
    # execute_tagsに含まれる単語が1つでも含まれていれば除去
    filtered_tags = [tag for tag in prompt_list if not any(execute_tag.lower() in tag.lower() for execute_tag in execute_tags)]
    return ", ".join(filtered_tags)
