import logging
from typing import Dict, List, Union

from src.task.task import Task, TaskType
from src.task.task_names import Tasks


def tasks_factory(task_names: Union[Dict, List[Tasks]]) -> List[Task]:
    """
    Factory method to create a list of Task objects from a dictionary of task names and their predictions.
    """
    tasks = []
    if isinstance(task_names, Dict):
        task_names = task_names.get("tasks")
        task_names = [list(task.keys())[0] for task in task_names]
    task_names = [
        Tasks(task_name) if isinstance(task_name, str) else task_name
        for task_name in task_names
    ]

    for task in task_names:
        match task:
            case Tasks.ALLOCINE:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.FQUAD:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="fquad",
                        task_type=TaskType.GENERATIVE,
                    )
                )
            case Tasks.GQNLI:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.PAWS_X:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.PIAF:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="fquad",
                        task_type=TaskType.GENERATIVE,
                    )
                )
            case Tasks.QFRBLIMP:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.QFRCOLA:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.SICKFR:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.STS22:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.XNLI:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.QFRCORE:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.FRCOE:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.QFRCORT:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.DACCORD:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.FRENCH_BOOLQ:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.MNLI_NINEELEVEN_FR_MT:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )

            case Tasks.RTE3_FRENCH:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.WINO_X_LM:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.WINO_X_MT:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.MULTIBLIMP:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.FRACAS:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.MMS:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.WSD:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="em",
                        task_type=TaskType.GENERATIVE,
                    )
                )
            case Tasks.LINGNLI:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.TIMELINE:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.LQLE:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="em",
                        task_type=TaskType.GENERATIVE,
                    )
                )
            case Tasks.QCCP:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="em",
                        task_type=TaskType.GENERATIVE,
                    )
                )
            case Tasks.QCCY:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="em",
                        task_type=TaskType.GENERATIVE,
                    )
                )
            case Tasks.QCCR:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="em",
                        task_type=TaskType.GENERATIVE,
                    )
                )
            case Tasks.PIQAFR:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case Tasks.PIQAQFR:
                tasks.append(
                    Task(
                        task_name=task.value,
                        metric="accuracy",
                        task_type=TaskType.INFERENCE,
                    )
                )
            case _:
                error = f"Unknown task {task.value}."
                logging.error(error)
                raise ValueError(error)
    return tasks
