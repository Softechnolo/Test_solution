from flask import Flask, request, jsonify
from recommend_fruits import recommend_fruits  


app = Flask(__name__)

@app.route('/recommend_fruits', methods=['POST'])
def get_recommended_fruits():
    try:
        data = request.get_json()
        client_answers = {
            "weekend_party": data.get("weekend_party"),
            "flavor": data.get("flavor"),
            "texture_dislike": data.get("texture_dislike"),
            "price_range": data.get("price_range")
        }

        recommended_fruits = recommend_fruits(**client_answers)
        return jsonify({"recommended_fruits": recommended_fruits})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
