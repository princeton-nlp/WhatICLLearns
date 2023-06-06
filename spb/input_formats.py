from abc import ABC, abstractmethod

from spb.input_example import InputExample
from spb.utils import add_period_for_lmbff

INPUT_FORMATS = {}

def register_input_format(format_class):
    INPUT_FORMATS[format_class.name] = format_class
    return format_class


class BaseInputFormat(ABC):
    name = None
    convert_to_true_false = False
    
    def format_input(self, example: InputExample):
        res = self._format_input(example=example)
        return res

    @abstractmethod
    def _format_input(self, example: InputExample) -> str:
        raise NotImplementedError



@register_input_format
class PlainInputFormat(BaseInputFormat):
    """
    This format uses the plain sentence as input.
    """
    name = 'plain'

    def _format_input(self, example: InputExample ) -> str:
        return ' '.join(example.tokens).strip()
    

@register_input_format
class ICLPlainInputFormat(BaseInputFormat):
    name = 'plain'

    def _format_input(self, example: InputExample ) -> str:
        return "Input: " + ' '.join(example.tokens).strip() + "\nOutput: "

###########################################################
#  
# Sentiment Classification Natural Formats
#
###########################################################


@register_input_format
class ICLSentimentNaturalInputFormat(BaseInputFormat):
    name = 'sentiment1'
    
    def _format_input(self, example: InputExample ) -> str:
        return ' '.join(example.tokens).strip() + "\nThe sentiment is "

@register_input_format
class ICLSentimentNaturalInputFormat1(BaseInputFormat):
    name = 'sentiment2'

    def _format_input(self, example: InputExample ) -> str:
        return ' '.join(example.tokens).strip() + "\nSentiment: "
    
@register_input_format
class ICLSentimentNaturalInputFormat2(BaseInputFormat):
    name = 'sentiment3'

    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + "\nThe sentiment of the text is  "

###########################################################
#  
# Non-NLI Minimal Formats
#
###########################################################

@register_input_format
class ClassificationMinimalInputFormat(BaseInputFormat):
    name = 'minimal1'

    def _format_input(self, example: InputExample ) -> str:
        sentence = ' '.join(example.tokens).rstrip()
        if sentence[-1] != ".":
            sentence = sentence + " ."
            
        return sentence + "\n"
    

@register_input_format
class ClassificationMinimalInputFormat2(BaseInputFormat):
    name = 'minimal2'

    def _format_input(self, example: InputExample ) -> str:
        sentence = ' '.join(example.tokens).rstrip()
        if sentence[-1] != ".":
            sentence = sentence + " ."
            
        return sentence + "\nLabel: "
    
@register_input_format
class ClassificationMinimalInputFormat3(BaseInputFormat):
    name = 'minimal3'

    def _format_input(self, example: InputExample ) -> str:
        sentence = ' '.join(example.tokens).rstrip()
        if sentence[-1] != ".":
            sentence = sentence + " ."
            
        return "Sentence: " + sentence + "\nLabel: "
    
    

###########################################################
#  
# General Hate Speech Natural Formats
#
###########################################################
    


@register_input_format
class ICLHateSpeechInputFormat(BaseInputFormat):
    name = 'hate_speech1'
    label_space = "y/n"
    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + "\nIs this hate speech? "
    
@register_input_format
class ICLHateSpeechInputFormat2(BaseInputFormat):
    name = 'hate_speech2'
    
    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + "\nIs the sentence hateful? "
    
@register_input_format
class ICLHateSpeechInputFormat3(BaseInputFormat):
    name = 'hate_speech3'
    convert_to_true_false = True
    
    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + "\nThe sentence contains hate speech. True or False?\nThe answer is "


###########################################################
#  
# Topic-Specific  Hate Speech Natural Formats
#
###########################################################
class TopicHateSpeechInputFormat(BaseInputFormat):
    name = 'hate_speech_topic1'
    topic = None

    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + f"\nIs this hate speech about {self.topic}? "
    
class TopicHateSpeechInputFormat2(BaseInputFormat):
    name = 'hate_speech_topic2'
    topic = None

    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + f"\nIs this hateful towards {self.topic}? "
    
class TopicHateSpeechInputFormat3(BaseInputFormat):
    name = 'hate_speech_topic3'
    topic = None
    convert_to_true_false = True

    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + f"\nThis is hate speech about {self.topic}. True or False?\nThe answer is "
    

@register_input_format
class ICLRaceHateSpeechInputFormat(TopicHateSpeechInputFormat):
    name = 'hate_speech_race1'
    topic = "race"

@register_input_format
class ICLGenderHateSpeechInputFormat(TopicHateSpeechInputFormat):
    name = 'hate_speech_gender1'
    topic = "gender"

@register_input_format
class ICLReligionHateSpeechInputFormat(TopicHateSpeechInputFormat):
    name = 'hate_speech_religion1'
    topic = "religion"
    
@register_input_format
class ICLNationalHateSpeechInputFormat(TopicHateSpeechInputFormat):
    name = 'hate_speech_national1'
    topic = "national origin"


@register_input_format
class ICLRaceHateSpeechInputFormat2(TopicHateSpeechInputFormat2):
    name = 'hate_speech_race2'
    topic = "race"

@register_input_format
class ICLGenderHateSpeechInputFormat2(TopicHateSpeechInputFormat2):
    name = 'hate_speech_gender2'
    topic = "gender"

@register_input_format
class ICLReligionHateSpeechInputFormat2(TopicHateSpeechInputFormat2):
    name = 'hate_speech_religion2'
    topic = "religion"
    
@register_input_format
class ICLNationalHateSpeechInputFormat2(TopicHateSpeechInputFormat2):
    name = 'hate_speech_national2'
    topic = "national origin"
    
    
@register_input_format
class ICLRaceHateSpeechInputFormat3(TopicHateSpeechInputFormat3):
    name = 'hate_speech_race3'
    topic = "race"

@register_input_format
class ICLGenderHateSpeechInputFormat3(TopicHateSpeechInputFormat3):
    name = 'hate_speech_gender3'
    topic = "gender"

@register_input_format
class ICLReligionHateSpeechInputFormat3(TopicHateSpeechInputFormat3):
    name = 'hate_speech_religion3'
    topic = "religion"
    
@register_input_format
class ICLNationalHateSpeechInputFormat3(TopicHateSpeechInputFormat3):
    name = 'hate_speech_national3'
    topic = "national origin"

###########################################################
#  
# Stance Detection Formats
#
###########################################################

class ICLTweetStanceInputFormat(BaseInputFormat):    
    stance_topic = None

    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + f"\nThe stance is {self.stance_topic}. True or False?\nThe answer is: "
    
class ICLTweetStanceInputFormat2(BaseInputFormat):
    stance_topic = None
    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + f"\nDoes the sentence express a {self.stance_topic} view?\n"
    
class ICLTweetStanceInputFormat3(BaseInputFormat):
    stance_topic = None
    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + f"\nIs the stance {self.stance_topic}?\n"
    
@register_input_format
class ICLTweetAtheismInputFormat(ICLTweetStanceInputFormat):
    name = 'tweet_atheism1'
    stance_topic = "atheist"
        
@register_input_format
class ICLTweetFeministInputFormat(ICLTweetStanceInputFormat):
    name = 'tweet_feminist1'
    stance_topic = "feminist"


@register_input_format
class ICLTweetAtheismInputFormat2(ICLTweetStanceInputFormat2):
    name = 'tweet_atheism2'
    stance_topic = "atheist"
        
@register_input_format
class ICLTweetFeministInputFormat2(ICLTweetStanceInputFormat2):
    name = 'tweet_feminist2'
    stance_topic = "feminist"


@register_input_format
class ICLTweetAtheismInputFormat3(ICLTweetStanceInputFormat3):
    name = 'tweet_atheism3'
    stance_topic = "atheist"
        
@register_input_format
class ICLTweetFeministInputFormat3(ICLTweetStanceInputFormat3):
    name = 'tweet_feminist3'
    stance_topic = "feminist"

    
###########################################################
#  
# Topic Detection Formats
#
###########################################################

@register_input_format
class TRECNaturalInputFormat(BaseInputFormat):
    name = 'topic1'
    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + "\nThe topic is "

@register_input_format
class TRECNaturalInputFormat2(BaseInputFormat):
    name = 'topic2'
    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + "\nThe sentence is about "

@register_input_format
class TRECNaturalInputFormat3(BaseInputFormat):
    name = 'topic3'
    def _format_input(self, example: InputExample ) -> str:
        sentence = add_period_for_lmbff(' '.join(example.tokens).rstrip())
        return sentence + "\nSentence topic: "
    
###########################################################
#  
# Paraphrase Detection Natural Formats
#
###########################################################
@register_input_format
class ParaphraseNaturalInputFormat(BaseInputFormat):
    name = 'paraphrase1'   
    convert_to_true_false = True

    def _format_input(self, example: InputExample ) -> str:
        return f"{example.sentence1}\nThe question is: {example.sentence2}\nTrue or False?\nThe answer is: "

@register_input_format
class ParaphraseNaturalInputFormat2(BaseInputFormat):
    name = 'paraphrase2'
    convert_to_true_false = True    

    def _format_input(self, example: InputExample ) -> str:
        return f"Sentence 1: {example.sentence1}\nSentence 2: {example.sentence2}\nThese sentences are paraphrases. True or False?\nThe answer is: "

@register_input_format
class ParaphraseNaturalInputFormat3(BaseInputFormat):
    
    name = 'paraphrase3'

    def _format_input(self, example: InputExample ) -> str:
        return f"Text: {example.sentence1}\nConsider this sentence: {example.sentence2}\nDoes it paraphrase the text?\n"

###########################################################
#  
# Entailment Detection Natural Formats
#
###########################################################
@register_input_format
class NLIEntailmentNaturalInputFormat(BaseInputFormat):
    name = 'nli_entailment1'
    convert_to_true_false = True
    def _format_input(self, example: InputExample ) -> str:
        return f"{example.sentence1}\nThe question is: {example.sentence2}\nTrue or False?\nThe answer is: "

@register_input_format
class NLIEntailmentNaturalInputFormat2(BaseInputFormat):
    name = 'nli_entailment2'

    def _format_input(self, example: InputExample ) -> str:
        return f"Hypothesis: {example.sentence1}\nPremise: {example.sentence2}\Do the sentences show entailment?\n"


@register_input_format
class NLIEntailmentNaturalInputFormat3(BaseInputFormat):
    
    name = 'nli_entailment3'

    def _format_input(self, example: InputExample ) -> str:
        return f"The hypothesis is: {example.sentence1}\The premise is: {example.sentence2}\nIs this entailment?\n"



###########################################################
#  
# NLI Detection Natural Formats
#
###########################################################
@register_input_format
class NLINaturalInputFormat(BaseInputFormat):
    name = 'nli1'
    convert_to_true_false = True

    def _format_input(self, example: InputExample ) -> str:
        return f"{example.sentence1}\nThe question is: {example.sentence2}\nTrue, False, or Unknown?\nThe answer is: "

@register_input_format
class NLINaturalInputFormat2(BaseInputFormat):
    
    name = 'nli2'

    def _format_input(self, example: InputExample ) -> str:
        return f"Hypothesis: {example.sentence1}\nPremise: {example.sentence2}\nGiven the premise, is the hypothesis true? Yes, No, or Unknown?\nThe answer is: "


@register_input_format
class NLINaturalInputFormat3(BaseInputFormat):
    
    name = 'nli3'
    convert_to_true_false = True

    def _format_input(self, example: InputExample ) -> str:
        return f"The hypothesis is: {example.sentence1}\nThe premise is: {example.sentence2}\nAccording to the premise, the hypothesis is true. True, False, or Unknown?\nThe answer is: "



###########################################################
#  
# NLI Minimal Formats #TODO
#
###########################################################

@register_input_format
class NLIMinimalInputFormat(BaseInputFormat):
    name = 'nli_minimal1'

    def _format_input(self, example: InputExample ) -> str:
        return f"{example.sentence1} [SEP] {example.sentence2}\n"
    
@register_input_format
class NLIMinimalInputFormat2(BaseInputFormat):
    name = 'nli_minimal2'

    def _format_input(self, example: InputExample ) -> str:
        return f"{example.sentence1} [SEP] {example.sentence2}\nLabel:"
    
@register_input_format
class NLIMinimalInputFormat3(BaseInputFormat):
    name = 'nli_minimal3'

    def _format_input(self, example: InputExample ) -> str:
        return f"Sentence 1: {example.sentence1}\nSentence 2: {example.sentence2}\nLabel: "
