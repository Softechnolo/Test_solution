
def recommend_fruits(weekend_party: str, flavor: str, texture_dislike: str, price_range: float) -> list:
    """
    Recommends fruits based on client's answers to the four questions.

    Args:
        weekend_party (str): "yes" if client goes out to party on weekends, else "no".
        flavor (str): Flavor preference ("cider", "sweet", or "waterlike").
        texture_dislike (str): Texture disliked ("smooth", "slimy", or "rough").
        price_range (float): Price range in dollars.

    Returns:
        list: List of recommended fruits.
    """
    allowed_fruits = ["oranges", "apples", "pears", "grapes", "watermelon", "lemon", "lime"]

    # Apply rules based on client's answers
    if weekend_party.lower() == "yes":
        allowed_fruits.extend(["apples", "pears", "grapes", "watermelon"])

    if flavor.lower() == "cider":
        allowed_fruits.extend(["apples", "oranges", "lemon", "lime"])
    elif flavor.lower() == "sweet":
        allowed_fruits.extend(["watermelon", "oranges"])
    elif flavor.lower() == "waterlike":
        allowed_fruits.remove("watermelon")

    if texture_dislike.lower() == "smooth":
        allowed_fruits.remove("pears")
    elif texture_dislike.lower() == "slimy":
        allowed_fruits = [fruit for fruit in allowed_fruits if fruit not in ["watermelon", "lime", "grapes"]]
    elif texture_dislike.lower() == "waterlike":
        allowed_fruits.remove("watermelon")

    if price_range < 3:
        allowed_fruits = [fruit for fruit in allowed_fruits if fruit not in ["lime", "watermelon"]]
    elif 4 < price_range < 7:
        allowed_fruits = [fruit for fruit in allowed_fruits if fruit not in ["pears", "apples"]]

    return allowed_fruits

# Example usage:
client_answers = {
    "weekend_party": "yes",
    "flavor": "cider",
    "texture_dislike": "smooth",
    "price_range": 5
}

recommended_fruits = recommend_fruits(**client_answers)
print("Recommended fruits:", recommended_fruits)
