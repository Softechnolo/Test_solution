from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def recommend_fruits(answers):
  rules = {
    'party': {'yes': ['apples', 'pears', 'grapes', 'watermelon']},
    'flavour': {'cider': ['apples', 'oranges', 'lemon', 'lime'], 'sweet': ['watermelon', 'orange'], 'waterlike': ['watermelon']},
    'texture': {'smooth': ['pears'], 'slimy': ['watermelon', 'lime', 'grape'], 'rough': []},
    'price': {'1': ['lime', 'watermelon'], '2': ['lime', 'watermelon'], '3': [], '4': ['pears', 'apples'], '5': ['pears', 'apples'], '6': [], '7': [], '8': [], '9': [], '10': []}
  }
  fruits = set(['oranges', 'apples', 'pears', 'grapes', 'watermelon', 'lemon', 'lime'])
  for key, answer in answers.items():
    if key in rules:
      for rule, remove_fruits in rules[key].items():
        if rule == answer:
          fruits -= set(remove_fruits)
  return list(fruits)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/recommend_fruits', methods=['POST'])
def recommend_fruits_api():
  answers = request.get_json()
  recommended_fruits = recommend_fruits(answers)
  return jsonify(recommended_fruits)

if __name__ == '__main__':
  app.run(debug=True)
