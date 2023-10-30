import requests
from bs4 import BeautifulSoup
import re
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate


def get_texts_by_url(url: str) -> str:
    """
    Extracts and preprocesses textual information from the web page
    :param url: URL of the web page
    :return: String with extracted text
    """
    try:
        response = requests.get(url)

        if response.status_code != 200:
            print('Error, please check the link')
            return ''

        content = response.text
        soup = BeautifulSoup(content, 'html.parser')

        # delete repeated space characters including linebreaks
        texts = re.sub(r'\n +', '\n', soup.get_text())
        texts = re.sub(r'\n+', '\n', texts)
        texts = re.sub(r' +', ' ', texts)
        texts = '\n'.join(list(map(lambda x: x.strip(), texts.split('\n'))))

    # some links may cause request errors
    except Exception as e:
        texts = ''
        print(e)

    return texts.strip()


def get_prompt(web_page: str) -> list:
    """
    Generates a prompt for the LLM
    :param web_page: preprocessed web page chunk
    :return: prompt for the LLM
    """
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You have to recognise furniture product entities given parsed web page"
                )
            ),
            HumanMessagePromptTemplate.from_template("""This is a parsed web page:
            
            {web_page}

            Extract all possible product names from this page and return them as a list of strings.
            Do not extract product categories.
            If the page doesn't contain any product names, return only word "None".
            Examples of product names are:
            "Lucca Bed Frame Fabric Gas Lift Storage - Grey King",
            "Boucle Stripe Cushion - Tan Peach",
            "Explorer 5 Dining Table" etc.
            """),
        ]
    )

    prompt = template.format_messages(web_page=web_page)

    return prompt
