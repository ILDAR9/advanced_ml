import os
import dask.dataframe as dd
import argparse
import json
import datetime

from upcampaign.datalib.connection import Engine
from upcampaign.utils.logging import create_logger
from upcampaign.campaign_flow import CampaignFlow


def create_engine(data_root_path: str) -> Engine:
    engine = Engine(tables={
        'receipts': dd.read_parquet(os.path.join(data_root_path, 'receipts.parquet')),
        'campaigns': dd.read_csv(os.path.join(data_root_path, 'campaigns.csv')),
        'client_profile': dd.read_csv(os.path.join(data_root_path, 'client_profile.csv')),
        'purchases': dd.read_parquet(os.path.join(data_root_path, 'purchases.parquet/')),
        'products': dd.read_csv(os.path.join(data_root_path, 'products.csv')),
    })
    return engine


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(
        description="help/guide/FAQ - <link>", prog="Run uplift campaign"
    )
    args_parser.add_argument("--run-id", required=True)
    args_parser.add_argument("--config", required=True)
    args_parser.add_argument("--system-config", required=True)
    args_parser.add_argument("--date-to", required=True)
    args_parser.add_argument('--silent', action='store_false', dest='verbose')
    args_parser.add_argument("-o", "--output")
    args = vars(args_parser.parse_args())

    with open(args['system_config'], 'r') as file:
        system_config = json.load(file)
    engine = create_engine(system_config['database']['root_path'])
    logger = create_logger('uplift-campaign')
    artifacts_root_path = system_config['artifacts_root_path']
    runs_root_path = system_config['runs_root_path']

    run_id = args.get('run_id', f"upcampaign_{datetime.datetime.now().strftime('%m%d_%H%M%S')}")
    with open(args['config'], 'r') as file:
        config = json.load(file)
    config['date_to'] = args.get('date_to', config['date_to'])

    flow = CampaignFlow(
        engine=engine,
        config=config,
        run_id=run_id,
        runs_root_path=runs_root_path,
        artifacts_root_path=artifacts_root_path,
        logger=logger
    )
    flow.run()

    output_path = args.get('output', None) 
    if output_path is not None:
        flow.result.to_csv(output_path, index=None)
