import pickle
import sklearn
import streamlit as st


model=pickle.load(open("spam.pkl", "rb"))
cv=pickle.load(open("vectorizer.pkl", "rb"))


def main():
    st.title("AI detects :red[spam] messages in emails ✉️")
    st.subheader(":green[Build With Streamlit & Python]", divider='rainbow')
    msg=st.text_input("Enter a Text: ")
    if st.button(":yellow[check]"):
        data=[msg]
        vect=cv.transform(data).toarray()
        prediction=model.predict(vect)
        result=prediction[0]
        if result==1:
            st.error("This is spam")
        else:
            st.success("This is not spam")

main()
