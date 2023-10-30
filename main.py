import argparse
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from product_extractor import product_extractor
from dotenv import load_dotenv
import os


load_dotenv()

API_KEY = os.getenv('API_KEY')
API_TYPE = os.getenv('API_TYPE')
API_VERSION = os.getenv('API_VERSION')
ENDPOINT = os.getenv('ENDPOINT')
LLM_DEPLOYMENT = os.getenv('LLM_DEPLOYMENT')

def main():
    # simple command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    args = parser.parse_args()
    url = args.url

    # splits web page into overlapping chunks
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size=10000,
                                          chunk_overlap=100)
    llm = AzureChatOpenAI(openai_api_base=ENDPOINT,
                          openai_api_version=API_VERSION,
                          deployment_name=LLM_DEPLOYMENT,
                          openai_api_key=API_KEY,
                          openai_api_type=API_TYPE)

    product_list = product_extractor(url=url, splitter=text_splitter, llm=llm)

    if not product_list:
        print('No products were found on this page')

    for product in product_list:
        print(product)


if __name__ == '__main__':
    main()
