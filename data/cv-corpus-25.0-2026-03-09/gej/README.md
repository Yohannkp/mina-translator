# *gej* &mdash; Mina (`gej`)

This datasheet is for cv-corpus-25.0-2026-03-09 of the Mozilla Common Voice *Scripted Speech* dataset for Mina [gej - `gej`]. The dataset contains 16773 clips representing 11.38 hours of recorded speech (11.14 hours validated) from 20 speakers, recorded from a text corpus of 3,188 sentences.

## Language

Mina (also called Gen, Gɛn gbe, Popo) is a Gbe language spoken primarily in southern Togo (Maritime Region) and in southern Benin (Mono Department). It is part of the Kwa branch of the Niger-Congo family. The language is used in daily communication, religious contexts, oral literature, and traditional expressions.

## Demographic information

The dataset includes the following self-declared age and gender distributions. A coverage summary is shown below each table.

### Gender

Self-declared gender information. The table shows clip and speaker counts with percentages. Speakers who did not declare a gender are listed as Unspecified. A dash (-) indicates zero.

| Code | Gender | Clips | Speakers |
|---|---|---|---|
| male_masculine | Male, masculine | - | - |
| female_feminine | Female, feminine | - | - |
| transgender | Transgender | - | - |
| non-binary | Non-binary | - | - |
| do_not_wish_to_say | Prefer not to say | - | - |
| - | Unspecified | 16,773 (100.0%) | 20 (100.0%) |

*Gender declared: 0 of 16,773 clips (0.0%), 0 of 20 speakers (0.0%)*

### Age

Self-declared age information. The table shows clip and speaker counts with percentages. Speakers who did not declare an age are listed as Unspecified. A dash (-) indicates zero.

| Code | Age | Clips | Speakers |
|---|---|---|---|
| teens | Teens | - | - |
| twenties | Twenties | - | - |
| thirties | Thirties | - | - |
| fourties | Fourties | - | - |
| fifties | Fifties | - | - |
| sixties | Sixties | - | - |
| seventies | Seventies | - | - |
| eighties | Eighties | - | - |
| nineties | Nineties | - | - |
| - | Unspecified | 16,773 (100.0%) | 20 (100.0%) |

*Age declared: 0 of 16,773 clips (0.0%), 0 of 20 speakers (0.0%)*

## Data splits for modelling

**Clip buckets**

| Bucket | Clips |
|---|---|
| Validated | 16,417 (97.9%) |
| Invalidated | 333 (2.0%) |
| Other | 23 (0.1%) |

**Training splits**

| Split | Clips |
|---|---|
| Train | 1,293 (7.9%) |
| Dev | 946 (5.8%) |
| Test | 949 (5.8%) |

*Training split coverage: 3,188 of 16,417 validated clips (19.4%)*

The dataset contains 16417 validated, 333 invalidated, and 23 unresolved clips. The average clip duration is 2.444 seconds.

## Text corpus

**Validated sentences:** 3,188

| Category | Count |
|---|---|
| Unvalidated sentences | - |
| Pending sentences | - |
| Rejected sentences | - |
| Reported sentences | - |

The corpus contains 3,188 sentences: 3,188 validated and 0 unvalidated (0 pending review, 0 rejected), with 0 reported for review.

### Writing system

Mina / Gen is a **tonal language**, using level tones:  

- High tone: marked with an acute accent (´)  
- Low tone: unmarked  

Nasalization is indicated via **n** following the vowel (e.g. *an, en*) rather than via diacritics.

#### Symbol table

Uppercase:  
```A B C D Ɖ E Ɛ F G GB Ɣ H X I J K KP L M N NY Ŋ Ɔ P S T U Ʋ V W Y Z  ```

Lowercase:  
```a b c d ɖ e ɛ f g gb ɣ h x i j k kp l m n ny ŋ ɔ p s t u ʋ v w y z  ```

### Sample

There follows a randomly selected sample of five sentences from the corpus.

1. *E sia nu.*
2. *Kotokuwoeadre.*
3. *Mi me te.*
4. *Anyigba.*
5. *Sosokpejesi.*

### Sources

The dataset has been compiled from:  

* Religious books (translations, liturgy)  
* Dictionaries of Mina / Gen  
* Common expressions collected from everyday life and fieldwork

| Source | Sentences |
|---|---|
| self | 3,188 (100.0%) |

### Fields

#### Clips

Each row of a `tsv` file represents a single audio clip, and contains the following information:

- `client_id` - hashed UUID of a given user
- `path` - relative path of the audio file
- `text` - supposed transcription of the audio
- `up_votes` - number of people who said audio matches the text
- `down_votes` - number of people who said audio does not match text
- `age` - age of the speaker[^1]
- `gender` - gender of the speaker[^1]
- `accents` - accents of the speaker[^1]
- `variant` - variant of the language[^1]
- `segment` - if sentence belongs to a custom dataset segment, it will be listed here
- `prompt_upvotes` - number of upvotes the sentence prompt received
- `prompt_reports` - number of reports the sentence prompt received
- `is_edited` - whether the clip's transcription has been edited

[^1]: For a full list of age, gender, and accent options, see the [demographics spec](https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts). These will only be reported if the speaker opted in to provide that information.

#### `validated_sentences.tsv`

The `validated_sentences.tsv` file contains one row per validated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `is_used` - whether the sentence is still in circulation for recording
- `clips_count` - number of clips recorded for this sentence

#### `unvalidated_sentences.tsv`

The `unvalidated_sentences.tsv` file contains one row per unvalidated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `up_votes` - number of upvotes the sentence received
- `down_votes` - number of downvotes the sentence received
- `status` - current status of the sentence (`pending` or `rejected`)

## Get involved

### Community links

* [Common Voice Mina / Gen (gej) page](https://commonvoice.mozilla.org/gej) *(link to be activated when available)*

- [Common Voice translators on Pontoon](https://pontoon.mozilla.org/gej/common-voice/contributors/)
- [Common Voice Communities](https://github.com/common-voice/common-voice/blob/main/docs/COMMUNITIES.md)

### Discussions

- [Common Voice on Matrix](https://chat.mozilla.org/#/room/#common-voice:mozilla.org)
- [Common Voice on Discourse](https://discourse.mozilla.org/t/about-common-voice-readme-first/17218)
- [Common Voice on Discord](https://discord.gg/9QTj9zwn)
- [Common Voice on Telegram](https://t.me/mozilla_common_voice)

### Contribute

- [Speak](https://commonvoice.mozilla.org/gej/speak)
- [Write](https://commonvoice.mozilla.org/gej/write)
- [Listen](https://commonvoice.mozilla.org/gej/listen)
- [Review](https://commonvoice.mozilla.org/gej/review)

## Acknowledgements

### Datasheet authors

* **Justin Bakoubolo**  
* **AGBOBLI, PhD**

* **Justin Bakoubolo** — supervised the compilation and coordination of the dataset.

* **Justin Bakoubolo** <[justin.bakoubolo@umbaji.org](mailto:justin.bakoubolo@umbaji.org)>

### Funding

This dataset was partially funded by the *Open Multilingual Speech Fund* managed by Mozilla Common Voice.

## Licence

This dataset is released under the [Creative Commons Zero (CC-0)](https://creativecommons.org/public-domain/cc0/) licence. By downloading this data you agree to not determine the identity of speakers in the dataset.
