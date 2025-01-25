# Scraped from https://huggingface.co/Helsinki-NLP
# Scrape code:
# console.log(Array.from(document.getElementsByClassName("text-md truncate font-mono text-black dark:group-hover/repo:text-yellow-500 group-hover/repo:text-indigo-600 text-smd")).map(element => element.innerHTML));

helsinki_models = [
  "Helsinki-NLP/opus-tatoeba-it-he",
  "Helsinki-NLP/opus-tatoeba-he-it",
  "Helsinki-NLP/opus-tatoeba-he-fr",
  "Helsinki-NLP/opus-tatoeba-fr-it",
  "Helsinki-NLP/opus-tatoeba-es-zh",
  "Helsinki-NLP/opus-mt-it-vi",
  "Helsinki-NLP/opus-mt-it-uk",
  "Helsinki-NLP/opus-mt-it-sv",
  "Helsinki-NLP/opus-mt-it-ms",
  "Helsinki-NLP/opus-mt-it-lt",
  "Helsinki-NLP/opus-mt-it-is",
  "Helsinki-NLP/opus-mt-it-fr",
  "Helsinki-NLP/opus-mt-it-es",
  "Helsinki-NLP/opus-mt-it-eo",
  "Helsinki-NLP/opus-mt-it-en",
  "Helsinki-NLP/opus-mt-it-de",
  "Helsinki-NLP/opus-mt-it-ca",
  "Helsinki-NLP/opus-mt-it-bg",
  "Helsinki-NLP/opus-mt-it-ar",
  "Helsinki-NLP/opus-mt-fr-zne",
  "Helsinki-NLP/opus-mt-fr-yo",
  "Helsinki-NLP/opus-mt-fr-yap",
  "Helsinki-NLP/opus-mt-fr-xh",
  "Helsinki-NLP/opus-mt-fr-wls",
  "Helsinki-NLP/opus-mt-fr-war",
  "Helsinki-NLP/opus-mt-fr-vi",
  "Helsinki-NLP/opus-mt-fr-ve",
  "Helsinki-NLP/opus-mt-fr-uk",
  "Helsinki-NLP/opus-mt-fr-ty",
  "Helsinki-NLP/opus-mt-fr-tw",
  "Helsinki-NLP/opus-mt-fr-tvl",
  "Helsinki-NLP/opus-mt-fr-tum",
  "Helsinki-NLP/opus-mt-fr-ts",
  "Helsinki-NLP/opus-mt-fr-tpi",
  "Helsinki-NLP/opus-mt-fr-to",
  "Helsinki-NLP/opus-mt-fr-tn",
  "Helsinki-NLP/opus-mt-fr-tll",
  "Helsinki-NLP/opus-mt-fr-tl",
  "Helsinki-NLP/opus-mt-fr-tiv",
  "Helsinki-NLP/opus-mt-fr-swc",
  "Helsinki-NLP/opus-mt-fr-sv",
  "Helsinki-NLP/opus-mt-fr-st",
  "Helsinki-NLP/opus-mt-fr-srn",
  "Helsinki-NLP/opus-mt-fr-sn",
  "Helsinki-NLP/opus-mt-fr-sm",
  "Helsinki-NLP/opus-mt-fr-sl",
  "Helsinki-NLP/opus-mt-fr-sk",
  "Helsinki-NLP/opus-mt-fr-sg",
  "Helsinki-NLP/opus-mt-fr-rw",
  "Helsinki-NLP/opus-mt-fr-run",
  "Helsinki-NLP/opus-mt-fr-ru",
  "Helsinki-NLP/opus-mt-fr-ro",
  "Helsinki-NLP/opus-mt-fr-rnd",
  "Helsinki-NLP/opus-mt-fr-pon",
  "Helsinki-NLP/opus-mt-fr-pl",
  "Helsinki-NLP/opus-mt-fr-pis",
  "Helsinki-NLP/opus-mt-fr-pap",
  "Helsinki-NLP/opus-mt-fr-pag",
  "Helsinki-NLP/opus-mt-fr-ny",
  "Helsinki-NLP/opus-mt-fr-nso",
  "Helsinki-NLP/opus-mt-fr-no",
  "Helsinki-NLP/opus-mt-fr-niu",
  "Helsinki-NLP/opus-mt-fr-mt",
  "Helsinki-NLP/opus-mt-fr-ms",
  "Helsinki-NLP/opus-mt-fr-mos",
  "Helsinki-NLP/opus-mt-fr-mh",
  "Helsinki-NLP/opus-mt-fr-mfe",
  "Helsinki-NLP/opus-mt-fr-lus",
  "Helsinki-NLP/opus-mt-fr-lue",
  "Helsinki-NLP/opus-mt-fr-lua",
  "Helsinki-NLP/opus-mt-fr-lu",
  "Helsinki-NLP/opus-mt-fr-loz",
  "Helsinki-NLP/opus-mt-fr-ln",
  "Helsinki-NLP/opus-mt-fr-lg",
  "Helsinki-NLP/opus-mt-fr-kwy",
  "Helsinki-NLP/opus-mt-fr-kqn",
  "Helsinki-NLP/opus-mt-fr-kg",
  "Helsinki-NLP/opus-mt-fr-iso",
  "Helsinki-NLP/opus-mt-fr-ilo",
  "Helsinki-NLP/opus-mt-fr-ig",
  "Helsinki-NLP/opus-mt-fr-id",
  "Helsinki-NLP/opus-mt-fr-hu",
  "Helsinki-NLP/opus-mt-fr-ht",
  "Helsinki-NLP/opus-mt-fr-hr",
  "Helsinki-NLP/opus-mt-fr-ho",
  "Helsinki-NLP/opus-mt-fr-hil",
  "Helsinki-NLP/opus-mt-fr-he",
  "Helsinki-NLP/opus-mt-fr-ha",
  "Helsinki-NLP/opus-mt-fr-guw",
  "Helsinki-NLP/opus-mt-fr-gil",
  "Helsinki-NLP/opus-mt-fr-gaa",
  "Helsinki-NLP/opus-mt-fr-fj",
  "Helsinki-NLP/opus-mt-fr-es",
  "Helsinki-NLP/opus-mt-fr-eo",
  "Helsinki-NLP/opus-mt-fr-en",
  "Helsinki-NLP/opus-mt-fr-el",
  "Helsinki-NLP/opus-mt-fr-efi",
  "Helsinki-NLP/opus-mt-fr-ee",
  "Helsinki-NLP/opus-mt-fr-de",
  "Helsinki-NLP/opus-mt-fr-crs",
  "Helsinki-NLP/opus-mt-fr-ceb",
  "Helsinki-NLP/opus-mt-fr-ca",
  "Helsinki-NLP/opus-mt-fr-bzs",
  "Helsinki-NLP/opus-mt-fr-bi",
  "Helsinki-NLP/opus-mt-fr-bg",
  "Helsinki-NLP/opus-mt-fr-ber",
  "Helsinki-NLP/opus-mt-fr-bem",
  "Helsinki-NLP/opus-mt-fr-bcl",
  "Helsinki-NLP/opus-mt-fr-ase",
  "Helsinki-NLP/opus-mt-fr-ar",
  "Helsinki-NLP/opus-mt-fr-af",
  "Helsinki-NLP/opus-mt-es-zai",
  "Helsinki-NLP/opus-mt-es-yua",
  "Helsinki-NLP/opus-mt-es-yo",
  "Helsinki-NLP/opus-mt-es-xh",
  "Helsinki-NLP/opus-mt-es-wls",
  "Helsinki-NLP/opus-mt-es-war",
  "Helsinki-NLP/opus-mt-es-vi",
  "Helsinki-NLP/opus-mt-es-ve",
  "Helsinki-NLP/opus-mt-es-uk",
  "Helsinki-NLP/opus-mt-es-tzo",
  "Helsinki-NLP/opus-mt-es-ty",
  "Helsinki-NLP/opus-mt-es-tw",
  "Helsinki-NLP/opus-mt-es-tvl",
  "Helsinki-NLP/opus-mt-es-tpi",
  "Helsinki-NLP/opus-mt-es-to",
  "Helsinki-NLP/opus-mt-es-tn",
  "Helsinki-NLP/opus-mt-es-tll",
  "Helsinki-NLP/opus-mt-es-tl",
  "Helsinki-NLP/opus-mt-es-swc",
  "Helsinki-NLP/opus-mt-es-st",
  "Helsinki-NLP/opus-mt-es-srn",
  "Helsinki-NLP/opus-mt-es-sn",
  "Helsinki-NLP/opus-mt-es-sm",
  "Helsinki-NLP/opus-mt-es-sl",
  "Helsinki-NLP/opus-mt-es-sg",
  "Helsinki-NLP/opus-mt-es-rw",
  "Helsinki-NLP/opus-mt-es-ru",
  "Helsinki-NLP/opus-mt-es-ro",
  "Helsinki-NLP/opus-mt-es-rn",
  "Helsinki-NLP/opus-mt-es-prl",
  "Helsinki-NLP/opus-mt-es-pon",
  "Helsinki-NLP/opus-mt-es-pl",
  "Helsinki-NLP/opus-mt-es-pis",
  "Helsinki-NLP/opus-mt-es-pap",
  "Helsinki-NLP/opus-mt-es-pag",
  "Helsinki-NLP/opus-mt-es-ny",
  "Helsinki-NLP/opus-mt-es-nso",
  "Helsinki-NLP/opus-mt-es-no",
  "Helsinki-NLP/opus-mt-es-nl",
  "Helsinki-NLP/opus-mt-es-niu",
  "Helsinki-NLP/opus-mt-es-mt",
  "Helsinki-NLP/opus-mt-es-mk",
  "Helsinki-NLP/opus-mt-es-mfs",
  "Helsinki-NLP/opus-mt-es-lus",
  "Helsinki-NLP/opus-mt-es-lua",
  "Helsinki-NLP/opus-mt-es-lt",
  "Helsinki-NLP/opus-mt-es-loz",
  "Helsinki-NLP/opus-mt-es-ln",
  "Helsinki-NLP/opus-mt-es-kg",
  "Helsinki-NLP/opus-mt-es-it",
  "Helsinki-NLP/opus-mt-es-iso",
  "Helsinki-NLP/opus-mt-es-is",
  "Helsinki-NLP/opus-mt-es-ilo",
  "Helsinki-NLP/opus-mt-es-ig",
  "Helsinki-NLP/opus-mt-es-id",
  "Helsinki-NLP/opus-mt-es-ht",
  "Helsinki-NLP/opus-mt-es-hr",
  "Helsinki-NLP/opus-mt-es-ho",
  "Helsinki-NLP/opus-mt-es-hil",
  "Helsinki-NLP/opus-mt-es-he",
  "Helsinki-NLP/opus-mt-es-ha",
  "Helsinki-NLP/opus-mt-es-guw",
  "Helsinki-NLP/opus-mt-es-gl",
  "Helsinki-NLP/opus-mt-es-gil",
  "Helsinki-NLP/opus-mt-es-gaa",
  "Helsinki-NLP/opus-mt-es-fr",
  "Helsinki-NLP/opus-mt-es-fj",
  "Helsinki-NLP/opus-mt-es-fi",
  "Helsinki-NLP/opus-mt-es-eu",
  "Helsinki-NLP/opus-mt-es-et",
  "Helsinki-NLP/opus-mt-es-es",
  "Helsinki-NLP/opus-mt-es-eo",
  "Helsinki-NLP/opus-mt-es-en",
  "Helsinki-NLP/opus-mt-es-el",
  "Helsinki-NLP/opus-mt-es-efi",
  "Helsinki-NLP/opus-mt-es-ee",
  "Helsinki-NLP/opus-mt-es-de",
  "Helsinki-NLP/opus-mt-es-da",
  "Helsinki-NLP/opus-mt-es-csn",
  "Helsinki-NLP/opus-mt-es-csg",
  "Helsinki-NLP/opus-mt-es-cs",
  "Helsinki-NLP/opus-mt-es-crs",
  "Helsinki-NLP/opus-mt-es-ceb",
  "Helsinki-NLP/opus-mt-es-ca",
  "Helsinki-NLP/opus-mt-es-bzs",
  "Helsinki-NLP/opus-mt-es-bi",
  "Helsinki-NLP/opus-mt-es-bg",
  "Helsinki-NLP/opus-mt-es-ber",
  "Helsinki-NLP/opus-mt-es-bcl",
  "Helsinki-NLP/opus-mt-es-ase",
  "Helsinki-NLP/opus-mt-es-ar",
  "Helsinki-NLP/opus-mt-es-af",
  "Helsinki-NLP/opus-mt-es-aed",
  "Helsinki-NLP/opus-mt-es-NORWAY",
  "Helsinki-NLP/opus-mt-en_el_es_fi-en_el_es_fi",
  "Helsinki-NLP/opus-mt-en-zlw",
  "Helsinki-NLP/opus-mt-en-zls",
  "Helsinki-NLP/opus-mt-en-zle",
  "Helsinki-NLP/opus-mt-en-zh",
  "Helsinki-NLP/opus-mt-en-xh",
  "Helsinki-NLP/opus-mt-en-vi",
  "Helsinki-NLP/opus-mt-en-urj",
  "Helsinki-NLP/opus-mt-en-ur",
  "Helsinki-NLP/opus-mt-en-umb",
  "Helsinki-NLP/opus-mt-en-uk",
  "Helsinki-NLP/opus-mt-en-ty",
  "Helsinki-NLP/opus-mt-en-tw",
  "Helsinki-NLP/opus-mt-en-tvl",
  "Helsinki-NLP/opus-mt-en-tut",
  "Helsinki-NLP/opus-mt-en-ts",
  "Helsinki-NLP/opus-mt-en-trk",
  "Helsinki-NLP/opus-mt-en-tpi",
  "Helsinki-NLP/opus-mt-en-toi",
  "Helsinki-NLP/opus-mt-en-to",
  "Helsinki-NLP/opus-mt-en-tn",
  "Helsinki-NLP/opus-mt-en-tll",
  "Helsinki-NLP/opus-mt-en-tl",
  "Helsinki-NLP/opus-mt-en-tiv",
  "Helsinki-NLP/opus-mt-en-ti",
  "Helsinki-NLP/opus-mt-en-tdt",
  "Helsinki-NLP/opus-mt-en-swc",
  "Helsinki-NLP/opus-mt-en-sw",
  "Helsinki-NLP/opus-mt-en-sv",
  "Helsinki-NLP/opus-mt-en-st",
  "Helsinki-NLP/opus-mt-en-ss",
  "Helsinki-NLP/opus-mt-en-sq",
  "Helsinki-NLP/opus-mt-en-sn",
  "Helsinki-NLP/opus-mt-en-sm",
  "Helsinki-NLP/opus-mt-en-sla",
  "Helsinki-NLP/opus-mt-en-sk",
  "Helsinki-NLP/opus-mt-en-sit",
  "Helsinki-NLP/opus-mt-en-sg",
  "Helsinki-NLP/opus-mt-en-sem",
  "Helsinki-NLP/opus-mt-en-sal",
  "Helsinki-NLP/opus-mt-en-rw",
  "Helsinki-NLP/opus-mt-en-run",
  "Helsinki-NLP/opus-mt-en-ru",
  "Helsinki-NLP/opus-mt-en-roa",
  "Helsinki-NLP/opus-mt-en-ro",
  "Helsinki-NLP/opus-mt-en-rnd",
  "Helsinki-NLP/opus-mt-en-rn",
  "Helsinki-NLP/opus-mt-en-pqw",
  "Helsinki-NLP/opus-mt-en-pqe",
  "Helsinki-NLP/opus-mt-en-poz",
  "Helsinki-NLP/opus-mt-en-pon",
  "Helsinki-NLP/opus-mt-en-pis",
  "Helsinki-NLP/opus-mt-en-phi",
  "Helsinki-NLP/opus-mt-en-pap",
  "Helsinki-NLP/opus-mt-en-pag",
  "Helsinki-NLP/opus-mt-en-om",
  "Helsinki-NLP/opus-mt-en-nyk",
  "Helsinki-NLP/opus-mt-en-ny",
  "Helsinki-NLP/opus-mt-en-nso",
  "Helsinki-NLP/opus-mt-en-nl",
  "Helsinki-NLP/opus-mt-en-niu",
  "Helsinki-NLP/opus-mt-en-nic",
  "Helsinki-NLP/opus-mt-en-ng",
  "Helsinki-NLP/opus-mt-en-mul",
  "Helsinki-NLP/opus-mt-en-mt",
  "Helsinki-NLP/opus-mt-en-mr",
  "Helsinki-NLP/opus-mt-en-mos",
  "Helsinki-NLP/opus-mt-en-ml",
  "Helsinki-NLP/opus-mt-en-mkh",
  "Helsinki-NLP/opus-mt-en-mk",
  "Helsinki-NLP/opus-mt-en-mh",
  "Helsinki-NLP/opus-mt-en-mg",
  "Helsinki-NLP/opus-mt-en-mfe",
  "Helsinki-NLP/opus-mt-en-map",
  "Helsinki-NLP/opus-mt-en-lus",
  "Helsinki-NLP/opus-mt-en-luo",
  "Helsinki-NLP/opus-mt-en-lun",
  "Helsinki-NLP/opus-mt-en-lue",
  "Helsinki-NLP/opus-mt-en-lua",
  "Helsinki-NLP/opus-mt-en-lu",
  "Helsinki-NLP/opus-mt-en-loz",
  "Helsinki-NLP/opus-mt-en-ln",
  "Helsinki-NLP/opus-mt-en-lg",
  "Helsinki-NLP/opus-mt-en-kwy",
  "Helsinki-NLP/opus-mt-en-kwn",
  "Helsinki-NLP/opus-mt-en-kqn",
  "Helsinki-NLP/opus-mt-en-kj",
  "Helsinki-NLP/opus-mt-en-kg",
  "Helsinki-NLP/opus-mt-en-jap",
  "Helsinki-NLP/opus-mt-en-itc",
  "Helsinki-NLP/opus-mt-en-it",
  "Helsinki-NLP/opus-mt-en-iso",
  "Helsinki-NLP/opus-mt-en-is",
  "Helsinki-NLP/opus-mt-en-ine",
  "Helsinki-NLP/opus-mt-en-inc",
  "Helsinki-NLP/opus-mt-en-ilo",
  "Helsinki-NLP/opus-mt-en-iir",
  "Helsinki-NLP/opus-mt-en-ig",
  "Helsinki-NLP/opus-mt-en-id",
  "Helsinki-NLP/opus-mt-en-hy",
  "Helsinki-NLP/opus-mt-en-hu",
  "Helsinki-NLP/opus-mt-en-ht",
  "Helsinki-NLP/opus-mt-en-ho",
  "Helsinki-NLP/opus-mt-en-hil",
  "Helsinki-NLP/opus-mt-en-hi",
  "Helsinki-NLP/opus-mt-en-he",
  "Helsinki-NLP/opus-mt-en-ha",
  "Helsinki-NLP/opus-mt-en-gv",
  "Helsinki-NLP/opus-mt-en-guw",
  "Helsinki-NLP/opus-mt-en-grk",
  "Helsinki-NLP/opus-mt-en-gmw",
  "Helsinki-NLP/opus-mt-en-gmq",
  "Helsinki-NLP/opus-mt-en-gl",
  "Helsinki-NLP/opus-mt-en-gil",
  "Helsinki-NLP/opus-mt-en-gem",
  "Helsinki-NLP/opus-mt-en-gaa",
  "Helsinki-NLP/opus-mt-en-ga",
  "Helsinki-NLP/opus-mt-en-fr",
  "Helsinki-NLP/opus-mt-en-fj",
  "Helsinki-NLP/opus-mt-en-fiu",
  "Helsinki-NLP/opus-mt-en-fi",
  "Helsinki-NLP/opus-mt-en-euq",
  "Helsinki-NLP/opus-mt-en-eu",
  "Helsinki-NLP/opus-mt-en-et",
  "Helsinki-NLP/opus-mt-en-es",
  "Helsinki-NLP/opus-mt-en-eo",
  "Helsinki-NLP/opus-mt-en-el",
  "Helsinki-NLP/opus-mt-en-efi",
  "Helsinki-NLP/opus-mt-en-ee",
  "Helsinki-NLP/opus-mt-en-dra",
  "Helsinki-NLP/opus-mt-en-de",
  "Helsinki-NLP/opus-mt-en-da",
  "Helsinki-NLP/opus-mt-en-cy",
  "Helsinki-NLP/opus-mt-en-cus",
  "Helsinki-NLP/opus-mt-en-cs",
  "Helsinki-NLP/opus-mt-en-crs",
  "Helsinki-NLP/opus-mt-en-cpp",
  "Helsinki-NLP/opus-mt-en-cpf",
  "Helsinki-NLP/opus-mt-en-chk",
  "Helsinki-NLP/opus-mt-en-cel",
  "Helsinki-NLP/opus-mt-en-ceb",
  "Helsinki-NLP/opus-mt-en-ca",
  "Helsinki-NLP/opus-mt-en-bzs",
  "Helsinki-NLP/opus-mt-en-bnt",
  "Helsinki-NLP/opus-mt-en-bi",
  "Helsinki-NLP/opus-mt-en-bg",
  "Helsinki-NLP/opus-mt-en-ber",
  "Helsinki-NLP/opus-mt-en-bem",
  "Helsinki-NLP/opus-mt-en-bcl",
  "Helsinki-NLP/opus-mt-en-bat",
  "Helsinki-NLP/opus-mt-en-az",
  "Helsinki-NLP/opus-mt-en-ar",
  "Helsinki-NLP/opus-mt-en-alv",
  "Helsinki-NLP/opus-mt-en-afa",
  "Helsinki-NLP/opus-mt-en-af",
  "Helsinki-NLP/opus-mt-en-aav",
  "Helsinki-NLP/opus-mt-en-ROMANCE",
  "Helsinki-NLP/opus-mt-en-CELTIC",
]

def get_clearly_formatted_langauge_directions():
    # Clearly formatted language directions only
    # Helsinki-NLP/opus-mt-tc-base-bat-zle is ignored
    # Helsinki-NLP/opus-mt-tc-fr-en is accepted due to clarity of source language being fr and target language being en
    language_directions = [model.split("Helsinki-NLP/opus-mt-")[1] for model in helsinki_models 
                           if (len(model.split("-"))<6 and 
                               "opus-mt" in model and 
                               len(model.split("_")) == 1)
                               and len(model) < len("Helsinki-NLP/opus-mt-src-trg")]
    return language_directions