class SettingsAccessor(object):
  def __getattr__(self, item):
    return getattr(active_settings, item)

class Standard(object):
  OVERWRITE_LOG = False
  IMMEDIATE_COMPUTE = False
  CHECK_VALIDITY = False
  RESOURCE_WARNINGS = False
  LOG_LEVEL_CONSOLE = "INFO"
  LOG_LEVEL_FILE = "DEBUG"
  DEFAULT_MOD_PATH = "{EXP_DIR}/models/{EXP}.mod"
  DEFAULT_LOG_PATH = "{EXP_DIR}/logs/{EXP}.log"

class Debug(Standard):
  OVERWRITE_LOG = True
  IMMEDIATE_COMPUTE = True
  CHECK_VALIDITY = True
  RESOURCE_WARNINGS = True
  LOG_LEVEL_CONSOLE = "DEBUG"
  LOG_LEVEL_FILE = "DEBUG"

class Unittest(Standard):
  OVERWRITE_LOG = True
  RESOURCE_WARNINGS = True
  LOG_LEVEL_CONSOLE = "WARNING"

active = Standard

aliases = {
  "settings.standard" : Standard,
  "settings.debug" : Debug,
  "settings.unittest" : Unittest,
}

def activate(settings_alias):
  global active
  active = aliases[settings_alias]