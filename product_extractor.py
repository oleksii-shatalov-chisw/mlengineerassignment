import re
from utils import get_texts_by_url, get_prompt


def product_extractor(url: str,
                      splitter,
                      llm) -> list[str] or None:
    """
    Extracts product names from web page using LLM

    :param url: URL of the specified web page
    :param splitter: CharacterTextSplitter instance
    :param llm: AzureChatOpenAI instance
    :return: list of products extracted from the web page
    """
    web_page_text = get_texts_by_url(url)
    text_chunks = splitter.split_text(web_page_text)

    llm_responses = []
    for chunk in text_chunks:
        prompt = get_prompt(chunk)
        llm_response = llm(prompt, temperature=0.01).content
        llm_responses.append(llm_response)

    # parse llm responses
    full_response = re.findall(r'(?<=\[)(.*?)(?=\])', '\n'.join(llm_responses))

    # return None if there are no product names
    if not full_response:
        return None

    product_list = []

    # get rid of whitespaces and double quotes
    for group in full_response:
        for product in group.split(','):
            product_list.append(product.strip()[1:-1])

    return product_list
