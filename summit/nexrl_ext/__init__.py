"""Summit's NexRL extensions.

Loaded onto the training VM and referenced from the compiled NexRL recipe via
`custom_trainer_module_path` / `custom_rollout_worker_module_path` — NexRL's
native extension points, so no monkeypatching is required.

(These modules import `nexrl`, which is only installed on the VM; importing
them on a laptop without NexRL will fail by design.)
"""
