from feast import FileSource, PushSource, FeatureView, Project, Field, Entity
from feast.types import Int64, Float64, Float32
from datetime import timedelta

enty = Entity(name='entity_id', join_keys=['entity_id'])

train_source = FileSource(
    name="train_data_source",
    path="/feast-data/fraud_detect/data/train.parquet",
    timestamp_field='event_timestamp',
    created_timestamp_column='created_timestamp'
)
train_push = PushSource(name='train_push_source', batch_source=train_source)

validate_source = FileSource(
    name="validate_data_source",
    path="/feast-data/fraud_detect/data/validate.parquet",
    timestamp_field='event_timestamp',
    created_timestamp_column='created_timestamp'
)
validate_push = PushSource(name='validate_push_source', batch_source=validate_source)

test_source = FileSource(
    name="test_data_source",
    path="/feast-data/fraud_detect/data/test.parquet",
    timestamp_field='event_timestamp',
    created_timestamp_column='created_timestamp'
)
test_push = PushSource(name='test_push_source', batch_source=test_source)

train_view = FeatureView(
    name="fraud_detection_train_view",
    entities=[enty],
    ttl=timedelta(days=1),
    schema=[
        Field(name='distance_from_last_transaction', dtype=Float64),
        Field(name='ratio_to_median_purchase_price', dtype=Float64),
        Field(name="used_chip", dtype=Float32),
        Field(name="used_pin_number", dtype=Float32),
        Field(name="online_order", dtype=Float32)
    ],
    source=train_push,
    online=True,
)
