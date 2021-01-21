def build_model_package(
    model_path: str, package_path: str, model_name: str, model_version: str, model_components: str
):
    """
    Add rules updating the model to the given path and package model

    :param model_path: A model path
    :param package_path: A package path
    :param model_name: A new model name
    :param model_version: A new model version
    :param model_components: A model components path
    """
    nlp = spacy.load(model_path)

    nlp.meta["name"] = model_name
    nlp.meta["version"] = str(model_version)

    ruler = EntityRuler(nlp, overwrite_ents=True)
    ruler.add_patterns(ruler_patterns)
    nlp.add_pipe(ruler)

    entity_matcher = EntityMatcher(nlp, matcher_patterns)
    nlp.add_pipe(entity_matcher)

    entity_custom = EntityCustom(nlp)
    nlp.add_pipe(entity_custom)

    nlp.to_disk(model_path)
    logger.info(f'Succesfully added rule based mathching at model: "{model_path}".')

    package(model_path, package_path)
    logger.info(f'Succesfully package at model: "{package_path}".')

    package_name = nlp.meta["lang"] + "_" + nlp.meta["name"]
    package_dir = package_name + "-" + nlp.meta["version"]
    package_base_path = os.path.join(package_path, package_dir, package_name)

    # Copio archivos con modulos custom al directorio pipe_components (no uso model_components para que sea fijo y siempre el mismo en nuestro componente, y asi poder desacoplarlo de cuando tengamos multiples clientes
    package_components_dir = "pipeline_components"
    files_src = ["entity_matcher.py", "entity_custom.py"]
    dest_component_dir = os.path.join(package_base_path, package_dir, package_components_dir)
    os.mkdir(dest_component_dir)
    for f in files_src:
        dest = os.path.join(dest_component_dir, f)
        orig = os.path.join(model_components, f)
        shutil.copyfile(orig, dest)

    new_code = f"""
        import os
        import importlib
        from spacy.language import Language
        def import_path(path):
            module_name = os.path.basename(path).replace('-', '_').replace('.','-')
            spec = importlib.util.spec_from_file_location(module_name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        
        dir =  os.fspath(Path(__file__).parent)
        moduloMatcher = import_path(dir + "/{package_dir}/{package_components_dir}/entity_matcher.py")
        moduloCustom = import_path(dir + "/{package_dir}/{package_components_dir}/entity_custom.py")
        
        Language.factories['entity_matcher'] = lambda nlp, **cfg: moduloMatcher.EntityMatcher(nlp, moduloMatcher.matcher_patterns,**cfg)
        Language.factories['entity_custom'] = lambda nlp, **cfg: moduloCustom.EntityCustom(nlp,**cfg)
    """

    insert_line = 6
    package_filename = "__init__.py"
    init_py = os.path.join(package_base_path, package_filename)
    with open(init_py, "r") as f:
        content = f.readlines()
    with open(init_py, "w") as f:
        for c in new_code.splitlines()[::-1]:
            content.insert(insert_line, c + "\n")
        f.writelines(content)
    logger.info("Succesfully write language factories to model")

def set_lr_and_beta1(train_config):
  # Adam settings and defaults
  if not "optimizer" in train_config:
      lr = 0.004
      beta1 = 0.9
  else:
      if "lr" in train_config["optimizer"]:
          lr = train_config["optimizer"]["lr"]
      else:
          lr = 0.004

      if "beta1" in train_config["optimizer"]:
          beta1 = train_config["optimizer"]["beta1"]
      else:
          beta1 = 0.9
  return lr, beta1

def set_dropout(train_config, FUNC_MAP):
  # FIXME NOT working with decaying
  if not "dropout" in train_config:
      return 0.2
  else:
      if type(train_config["dropout"]) == int or type(train_config["dropout"]) == float:
          return train_config["dropout"]
      else:
          d = train_config["dropout"]
          return FUNC_MAP[d.pop("f")](d["from"], d["to"], d["rate"])

def set_batch_size(train_config, FUNC_MAP):
  if not "batch_size" in train_config:
      return 4, ()
  else:
      if type(train_config["batch_size"]) == int or type(train_config["batch_size"]) == float:
          return train_config["batch_size"], ()
      else:
          b = train_config["batch_size"]
          return (FUNC_MAP[b.pop("f")], (b["from"], b["to"], b["rate"]))

def save_state_history(
    state,
    numero_losses,
    f_score,
    recall_score,
    precision_score,
    per_type_score,
    val_f_score,
    val_recall_score,
    val_precision_score,
    val_per_type_score,
    learn_rate,
    num_batches,
    dropout,
):
  state["history"]["ner"].append(numero_losses)
  state["history"]["f_score"].append(f_score)
  state["history"]["recall"].append(recall_score)
  state["history"]["precision"].append(precision_score)
  state["history"]["per_type_score"].append(per_type_score)

  # validation
  state["history"]["val_f_score"].append(val_f_score)
  state["history"]["val_recall"].append(val_recall_score)
  state["history"]["val_precision"].append(val_precision_score)
  state["history"]["val_per_type_score"].append(val_per_type_score)

  state["history"]["lr"].append(learn_rate)
  state["history"]["batches"].append(num_batches)
  state["history"]["dropout"].append(dropout)


def run_command_with_timer(*args):
  """
  Calculate the time spend to run command

  :param *args: run command
  """
  begin_time = datetime.datetime.now()
  logger.info("####START####")
  logger.info(f"Start process {begin_time} ")
  subprocess.call(args[0], shell=True)
  end = datetime.datetime.now()
  logger.info(f"End {end} to process ")
  logger.info(f"Spend {end-begin_time} to process ")