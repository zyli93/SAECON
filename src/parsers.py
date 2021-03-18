'''
Functions that parse the annotated data that is being used in this project. The
annotated dataset are the following:

1. `Li Dong <http://goo.gl/5Enpu7>`_ which links to :py:func:`tdparse.parsers.dong`
2. Semeval parser
'''
import json
import os
import re
import xml.etree.ElementTree as ET

from data_types import Target, TargetCollection


def _semeval_extract_data(sentences, file_name, conflict=False,
                          sentence_ids_skip=None):
    '''
    :param sentences: A `sentences` named element
    :param file_name: Name of the file being parsed
    :param conflict: Determine if to keep the target data that has a conflict \
    sentiment label.
    :param sentence_ids_skip: IDs of sentences that should be skipped
    :type sentences: xml.etree.ElementTree.Element
    :type file_name: String
    :type conflict: bool. Defailt False
    :type sentence_ids_skip: list. Default None
    :returns: A TargetCollection containing Target instances.
    :rtype: TargetCollection
    '''

    # Converts the sentiment tags from Strings to ints
    sentiment_mapper = {'conflict' : -2, 'negative' : -1,
                        'neutral' : 0, 'positive' : 1}

    def extract_aspect_terms(aspect_terms, sentence_id):
        '''
        :param aspect_terms: An aspectTerms element within the xml tree
        :param sentence_id: Id of the sentence that the aspects came from.
        :type aspect_terms: xml.etree.ElementTree.Element
        :type sentence_id: String
        :returns: A list of dictioanries containg id, span, sentiment and \
        target
        :rtype: list
        '''

        aspect_terms_data = []
        for index, aspect_term in enumerate(aspect_terms):
            aspect_term = aspect_term.attrib
            aspect_term_data = {}
            sentiment = sentiment_mapper[aspect_term['polarity']]
            if sentiment == -2 and not conflict:
                continue
            aspect_id = '{}{}'.format(sentence_id, index)
            aspect_term_data['target_id'] = aspect_id
            if 'term' in aspect_term:
                aspect_term_data['target'] = aspect_term['term']
            elif 'target' in aspect_term:
                aspect_term_data['target'] = aspect_term['target']
            else:
                raise KeyError('There is no `target` attribute in the opinions '\
                               'element {}'.format(aspect_term))
            aspect_term_data['sentiment'] = sentiment
            aspect_term_data['spans'] = [(int(aspect_term['from']),
                                          int(aspect_term['to']))]
            aspect_term_data['sentence_id'] = sentence_id
            # If the target is NULL then there is no target
            if aspect_term_data['target'] == 'NULL':
                continue
            aspect_terms_data.append(aspect_term_data)
        return aspect_terms_data

    def add_text(aspect_data, text):
        '''
        :param aspect_data: A list of dicts containing `span`, `target` and \
        `sentiment` keys.
        :param text: The text of the sentence that is associated to all of the \
        aspects in the aspect_data list
        :type aspect_data: list
        :type text: String
        :returns: The list of dicts in the aspect_data parameter but with a \
        `text` key with the value that the text parameter contains
        :rtype: list
        '''

        for data in aspect_data:
            data['text'] = text
        return aspect_data

    all_aspect_term_data = TargetCollection()
    for sentence in sentences:
        aspect_term_data = None
        text_index = None
        sentence_id = file_name + sentence.attrib['id']
        # Allow the parser to skip certain sentences
        if sentence_ids_skip is not None:
            if sentence.attrib['id'] in sentence_ids_skip:
                continue
        for index, data in enumerate(sentence):
            if data.tag == 'sentence':
                raise Exception(sentence.attrib['id'])
            if data.tag == 'text':
                text_index = index
            elif data.tag == 'aspectTerms' or data.tag == 'Opinions':
                aspect_term_data = extract_aspect_terms(data, sentence_id)
        if aspect_term_data is None:
            continue
        if text_index is None:
            raise ValueError('A semeval sentence should always have text '\
                             'semeval file {} sentence id {}'\
                             .format(file_name, sentence.attrib['id']))
        sentence_text = sentence[text_index].text
        aspect_term_data = add_text(aspect_term_data, sentence_text)
        for aspect in aspect_term_data:
            sent_target = Target(**aspect)
            all_aspect_term_data.add(sent_target)
    return all_aspect_term_data


def semeval_15_16(file_path, sep_16_from_15=False):
    '''
    Parser for the SemEval 2015 and 2016 datasets.

    :param file_path: File path to the semeval 2014 data
    :param sep_16_from_15: Ensure that the test sets of semeval 2016 is complete \
    seperate from the semeval test set of 2015
    :type file_path: String
    :type sep_16_from_15: bool. Default False
    :returns: A TargetCollection containing Target instances.
    :rtype: TargetCollection
    '''

    file_path = os.path.abspath(file_path)
    file_name, _ = os.path.splitext(os.path.basename(file_path))

    tree = ET.parse(file_path)
    reviews = tree.getroot()
    all_aspect_term_data = []
    if reviews.tag != 'Reviews':
        raise ValueError('The root of all semeval 15/16 xml files should '\
                         'be reviews and not {}'\
                         .format(reviews.tag))
    for review in reviews:
        review_id = review.attrib['rid']
        for sentences in review:
            if sep_16_from_15:
                ids_to_skip = ["en_SnoozeanAMEatery_480032670:4"]
                review_targets = _semeval_extract_data(sentences, file_name,
                                                       sentence_ids_skip=ids_to_skip)
                all_aspect_term_data.extend(review_targets.data())
            else:
                review_targets = _semeval_extract_data(sentences, file_name).data()
                all_aspect_term_data.extend(review_targets)
    return TargetCollection(all_aspect_term_data)

def semeval_14(file_path, conflict=False):
    '''
    Parser for the SemEval 2014 datasets.

    :param file_path: File path to the semeval 2014 data
    :param conflict: determine if to include the conflict sentiment value
    :type file_path: String
    :type conflict: bool. Default False.
    :returns: A TargetCollection containing Target instances.
    :rtype: TargetCollection
    '''
    file_path = os.path.abspath(file_path)
    file_name, _ = os.path.splitext(os.path.basename(file_path))

    tree = ET.parse(file_path)
    sentences = tree.getroot()
    if sentences.tag != 'sentences':
        raise ValueError('The root of all semeval xml files should '\
                         'be sentences and not {}'\
                         .format(sentences.tag))
    return _semeval_extract_data(sentences, file_name, conflict=conflict)