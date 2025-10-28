import Config

config :blake3_ex, Blake3.Native,
  mode: :release,
  features: MixBlake3.Project.config_features()

import_config("#{Mix.env()}.exs")
