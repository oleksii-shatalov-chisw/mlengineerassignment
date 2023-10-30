import json
from product_extractor import product_extractor
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os


load_dotenv()

API_KEY = os.getenv('API_KEY')
API_TYPE = os.getenv('API_TYPE')
API_VERSION = os.getenv('API_VERSION')
ENDPOINT = os.getenv('ENDPOINT')
LLM_DEPLOYMENT = os.getenv('LLM_DEPLOYMENT')



def get_product_data(json_file: str) -> tuple[list[str], list[list[str]]]:
    """
    Read dataset presented as json file and get list of URLs and ground-truth products
    :param json_file: path to the file
    :return: list of URLs and list of ground-truth products for each URL
    """

    with open(json_file, encoding='utf-8', mode='r') as file:
        products_data = json.load(file)

    urls = []
    products_lists = []
    for item in products_data:
        url = tuple(item.items())[0][0]
        products = tuple(item.items())[0][1]
        urls.append(url)
        products_lists.append(list(map(lambda string: string.lower(), products)))
    return urls, products_lists


def compute_metrics(url_list: list[str],
                    true_products_lists: list[list[str]],
                    splitter,
                    llm) -> tuple[float]:
    """
    Automatically compute precision, recall and f1 score by comparing LLM output and ground-truth product names
    :param url_list: List of test URLs
    :param true_products: List of ground-truth products extracted manually from each URL
    :param splitter: CharacterTextSplitter instance
    :param llm: AzureChatOpenAI instance
    :return: precision, recall and f1-score
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0


    with open('results.txt', mode='w', encoding='utf_8') as file:
        for i in tqdm(range(len(url_list))):
            url = url_list[i]

            file.write(f'URL: {url}\n')

            extracted_products_list = list(map(lambda string: string.lower(),
                                               product_extractor(url,
                                                                 splitter,
                                                                 llm)
                                               ))

            file.write(f'Extracted products: {str(extracted_products_list)}\n')
            file.write(f'True products: {str(true_products_lists[i])}\n\n')

            for extracted_product in extracted_products_list:
                if extracted_product in true_products_lists[i]:
                    true_positives += 1
                else:
                    false_positives += 1

            for k in range(len(true_products_lists[i])):
                if true_products_lists[i][k] not in extracted_products_list:
                    false_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def main():
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size=10000,
                                          chunk_overlap=100)
    llm = AzureChatOpenAI(openai_api_base=ENDPOINT,
                          openai_api_version=API_VERSION,
                          deployment_name=LLM_DEPLOYMENT,
                          openai_api_key=API_KEY,
                          openai_api_type=API_TYPE)

    urls, true_products_lists = get_product_data('data/scraped_products.json')

    precision, recall, f1 = compute_metrics(url_list=urls,
                                            true_products_lists=true_products_lists,
                                            splitter=text_splitter,
                                            llm=llm)

    print(f'Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}')


if __name__ == "__main__":
    main()
