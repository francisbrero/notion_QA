import streamlit as st
import time



def add_sidebar(st):
    """Adds the sidebar to the streamlit app."""
    with st.sidebar:

        st.markdown("# Hi, I am Nygel 🐈")

        st.markdown(
            """
            I am an AI-powered Support Agent. 
            
            I've been trained on MadKudu's Notion workspace. More specifically I've read all of the support runbooks.
            
            I can give you step by step instructions on how to solve a problem. I can also answer general questions about MadKudu.
            
            Just ask me anything!
        """)

        st.divider()

        st.markdown("""What makes me special?""")    

        with st.expander(' I was trained on Support Runbooks.'):
            st.markdown("""I have read hundreds of pages in [Notion](https://www.notion.so/madkudu/Support-runbooks-d2a894351f944fc5b4abb9f29f30b4a4)
                        ... yeah we have a few...
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("Made with ❤️ by the people at [MadKudu](https://madkudu.com)")


        with st.expander('Source'):
            source = """
            [Github Repo](https://github.com/francisbrero/notion_QA)
            """
            st.markdown(source, unsafe_allow_html=True)

        disclaimer = '<p style="font-size: 10px;">This LLM can make mistakes. Consider checking important information.</p>'
        st.markdown(disclaimer, unsafe_allow_html=True)

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


# create a function to format the sources as a link
# we will get the id from the metadata of the retrieved document as described in the example below:
# if sources = "notion_data/support_runbook/Support runbooks d2a894351f944fc5b4abb9f29f30b4a4/User cannot deploy model from the Studio dd2630484a334e159dd9bf07086824ad.md" then the id is dd2630484a334e159dd9bf07086824ad
prefix = 'https://www.notion.so/madkudu/'

def format_sources(sources):
    formatted_sources = ""
    if sources is None:
        return formatted_sources
    source_id = sources.split('/')[-1].split('.')[0].replace(' ','-')
    formatted_sources = prefix + source_id
    return formatted_sources