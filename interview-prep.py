import speech_recognition as sr
import pyttsx3
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import google.generativeai as palm
import numpy as np
# Set up the listener
listener = sr.Recognizer()
engine = pyttsx3.init()

# Set up the talk function for audio output
def talk(text):
    engine.say(text)
    engine.runAndWait()

def palm_api():
    Field_Selection = input("Please Enter your Stream/Field you are preparing for: ")
    API_KEY = 'AIzaSyDbm_oPoOm6WNVq9L14F4U2If5bUHgsG6U'  # Replace with your actual Palm API key
    palm.configure(api_key=API_KEY)
    model_id = 'models/text-bison-001'
    
    completion = palm.generate_text(
        model=model_id,
        prompt=f"Provide 10 Questions only for {Field_Selection} Interview Preparation.",
        temperature=0.99,
        max_output_tokens=800,
        candidate_count=1
    )
    
    outputs = [output['output'] for output in completion.candidates]
    questions = outputs[0].split('?')
    questions_list = [question.strip() for question in questions if question.strip()]
    
    return questions_list

def get_user_responses(questions_list):
    user_responses = []
    asked_questions = []

    for question in questions_list:
        asked_questions.append(question)
        talk(question)  # Speak the question
        
        try:
            with sr.Microphone() as source:
                print("Please answer the question:")
                audio = listener.listen(source)
                user_answer = listener.recognize_google(audio)
                print("Your answer:", user_answer)
                user_responses.append(user_answer)
        except sr.UnknownValueError:
            print("Speech recognition couldn't understand your response. Please type your answer:")
            user_answer = input()
            user_responses.append(user_answer)

    return asked_questions, user_responses

def calculate_similarity(reference_answers, user_responses):
    vectorizer = CountVectorizer().fit_transform(reference_answers + user_responses)
    vectors = vectorizer.toarray()
    reference_vector = vectors[:len(reference_answers)]
    user_response_vectors = vectors[len(reference_answers):]
    
    similarity_scores = cosine_similarity(reference_vector, user_response_vectors)
    average_similarity = np.mean(similarity_scores) * 100
    return average_similarity

def main():
    reference_answers = palm_api()
    asked_questions, user_responses = get_user_responses(reference_answers)

    similarity_score = calculate_similarity(reference_answers, user_responses)
    print(f"Your similarity score is: {similarity_score:.2f}%")

    save_to_file('asked_questions.txt', asked_questions)
    save_to_file('reference_answers.txt', reference_answers)
    save_to_file('user_responses.txt', user_responses)

def save_to_file(filename, content):
    with open(filename, 'w') as file:
        for line in content:
            file.write(line + '\n')

if __name__ == "__main__":
    main()
