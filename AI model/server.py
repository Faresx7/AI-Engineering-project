from flask import Flask, render_template, request, jsonify
import base as b

app = Flask(__name__)
print('===========================================================model is preparing===========================================================')
model = b.AIModel()
print('===========================================================model is ready...===========================================================')

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/chat', methods = ["POST"])
def generate():

    try:
        data = request.get_json()

        if not data or "prompt" not in data:
            return jsonify({'error':'No message have been sent!'}), 400

        prompt = data['prompt']
        model_response = model.generate(prompt)

        return jsonify({'response':model_response})

    except Exception as e: 
        return jsonify({'error':str(e)}), 500


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True, use_reloader = False)