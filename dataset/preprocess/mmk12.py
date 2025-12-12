from dataset.const import AnswerType, ProblemType


def preprocess_mmk12(sample):
    sample["source"] = "mmk12"
    sample["images"] = [sample.pop("image")]
    sample["problem"] = "<image>" + sample.pop("question")
    sample["problem_type"] = str(ProblemType.GENERAL)
    sample["answer_type"] = str(AnswerType.MATH_EXPRESSIONS)

    del sample["subject"]

    return sample

