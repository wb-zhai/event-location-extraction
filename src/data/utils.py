from gliner2.training.data import TrainingDataset


def collect_schema(datasets: list[TrainingDataset]) -> tuple[list[str], list[str]]:
    event_types = sorted(
        {
            event_type
            for dataset in datasets
            for example in dataset
            for event_type in example.entities.keys()
        }
    )
    argument_types = sorted(
        {
            relation.name
            for dataset in datasets
            for example in dataset
            for relation in example.relations
        }
    )
    return event_types, argument_types
