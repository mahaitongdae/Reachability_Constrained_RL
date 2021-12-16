from ray.rllib.agents.sac import SACTrainer

def try_ray_sac():
    config = {"env":"Pendulum-v0",
              "num_workers": 4,
              "framework": "tf",
              "evaluation_num_workers": 1,
              "evaluation_config": {
                  "render_env": True,
              }}
    trainer = SACTrainer(config=config)
    for _ in range(3):
        print(trainer.train())

    trainer.evaluate()

if __name__ == '__main__':
    try_ray_sac()