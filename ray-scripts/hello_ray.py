import ray

@ray.remote
def hello_ray():
    return "hello ray"


# Automatically connect to the running Ray cluster.
ray.init()
print(ray.get(hello_ray.remote()))