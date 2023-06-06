from abc import ABC, abstractmethod
from spb.input_example import InputExample

OUTPUT_FORMATS = {}

def register_output_format(format_class):
    OUTPUT_FORMATS[format_class.name] = format_class
    return format_class

class BaseOutputFormat(ABC):
    name = None
    link_str = "\nOutput: "
    seed = None

    @abstractmethod
    def format_output(self, example: InputExample, prompt_output=None) -> str:
        """
        Format output in augmented natural language.
        """
        raise NotImplementedError

    @abstractmethod
    def run_inference(self, example: InputExample, output_sentence: str):
        """
        Process an output sentence to extract whatever information the task asks for.
        """
        raise NotImplementedError

@register_output_format
class ClassificationFormat(BaseOutputFormat):
    """
    Output format for Classification datasets
    """
    name = 'classification'

    def format_output(self, example: InputExample, label_dict=None) -> str:
        """
        Get output in augmented natural language, for example:
        [belief] hotel price range cheap , hotel type hotel , duration two [belief]
        """
        return example.gt_label_name

    def run_inference(self, example: InputExample, output_sentence: str):
        """
        Process an output sentence to extract the predicted belief.
        """
        return output_sentence.strip()
