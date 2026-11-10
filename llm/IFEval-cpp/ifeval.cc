#include "ifeval.h"

#include <fstream>
#include <iostream>

namespace mlperf {
namespace mobile {

namespace {

std::string remove_special_tokens(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    if (s[i] == '<') {
      std::size_t j = i + 1;
      while (j < s.size() && s[j] != '<' && s[j] != '>' &&
             !std::isspace(static_cast<unsigned char>(s[j])))
        ++j;
      if (j < s.size() && s[j] == '>') {
        std::string piece = s.substr(i, j - i + 1);
        if (piece == "<pad>" || piece == "<eos>" || piece == "<bos>" ||
            piece == "<mask>" ||
            (piece.size() > 3 &&
             (piece.compare(0, 2, "<|") == 0 ||
              piece.compare(piece.size() - 2, 2, "|>") == 0))) {
          i = j + 1;
          continue;
        }
      }
    }
    out.push_back(s[i]);
    ++i;
  }
  return out;
}

}  // namespace

IFEval::IFEval(const std::string& input_tfrecord)
   {

       auto input_val = crow::json::load(input_tfrecord);
       if (!input_val) std::cout << "Could not parse input!" << std::endl;
       auto input = input_val.as_list();

  // Load all TFRecord samples into memory
  // NOTE this can be moved to LoadSamplesToRam, but will cause delays between
  // queries due to IO reads
  for (size_t i = 0; i < input.size(); i++) {
        std::cout << "reading record " << std::to_string(i+1) << '/' << std::to_string(input.size()) << std::endl;
    auto record = input[i];
    int key = record["key"].as_i();
    std::string prompt = record["prompt"].as_string();
    std::string response =
        remove_special_tokens(record["response"].as_string());
    auto instructions = BuildInstructions(record);

    //std::string input_formatted = FormatLlamaUserPrompt(prompt);
    //std::vector<int> input_tokens;
    //sp_processor->Encode(input_formatted.c_str(), &input_tokens).ok();

    auto sample = std::make_unique<ifeval::Sample>();
    sample->key = key;
    sample->prompt = prompt;
    sample->response = response;
    //sample->input_tokens = input_tokens;
    sample->instructions = std::move(instructions);

    samples_.push_back(std::move(sample));
    sample_output_tokens_.push_back(std::vector<int>());

    used_sample_ids_.emplace(i);
  }
  std::cout << "Record reading complete!" << std::endl;
}

bool IFEval::ComputeSampleAccuracy(const int sample_idx, ifeval::Accuracy& accuracy) {
  std::string prediction = samples_[sample_idx]->response;
  bool is_prompt_correct_loose = true;
  bool is_prompt_correct_strict = true;
  std::vector<bool> loose_res;
  std::vector<bool> strict_res;
  for (const auto& instruction : samples_[sample_idx]->instructions) {
    bool is_correct_loose = instruction->IsFollowed(prediction, true);
    bool is_correct_strict = instruction->IsFollowed(prediction, false);

    accuracy.instruction_total++;
    accuracy.instruction_correct_loose += is_correct_loose ? 1 : 0;
    loose_res.push_back(is_correct_loose);
    accuracy.instruction_correct_strict += is_correct_strict ? 1 : 0;
    strict_res.push_back(is_correct_strict);

    is_prompt_correct_loose =
        is_prompt_correct_loose ? is_correct_loose : false;
    is_prompt_correct_strict =
        is_prompt_correct_strict ? is_correct_strict : false;
  }

  accuracy.prompt_total++;
  accuracy.prompt_correct_loose += is_prompt_correct_loose ? 1 : 0;
  accuracy.prompt_correct_strict += is_prompt_correct_strict ? 1 : 0;

  samples_[sample_idx]->status_loose = loose_res;
  samples_[sample_idx]->status_strict = strict_res;

  return true;
}

float IFEval::ComputeAccuracy() {
  float instruction_loose_accuracy;
  float instruction_strict_accuracy;
  float prompt_loose_accuracy;
  float prompt_strict_accuracy;
  ifeval::Accuracy accuracy;

  for (auto sample_id : used_sample_ids_) {
    std::cout << "Computing accuracy for sample " << std::to_string(sample_id) << "..." << std::endl;
    ComputeSampleAccuracy(sample_id, accuracy);
  }

  instruction_loose_accuracy =
      accuracy.instruction_total > 0
          ? static_cast<float>(accuracy.instruction_correct_loose) /
                accuracy.instruction_total
          : 0.0f;
  instruction_strict_accuracy =
      accuracy.instruction_total > 0
          ? static_cast<float>(accuracy.instruction_correct_strict) /
                accuracy.instruction_total
          : 0.0f;
  prompt_loose_accuracy =
      accuracy.prompt_total > 0
          ? static_cast<float>(accuracy.prompt_correct_loose) /
                accuracy.prompt_total
          : 0.0f;
  prompt_strict_accuracy =
      accuracy.prompt_total > 0
          ? static_cast<float>(accuracy.prompt_correct_strict) /
                accuracy.prompt_total
          : 0.0f;

  std::cout << "Instruction-level loose-accuracy: " << std::to_string(instruction_loose_accuracy) << std::endl;
  std::cout << "Instruction-level strict-accuracy: " << std::to_string(instruction_strict_accuracy) << std::endl;
  std::cout << "Prompt-level loose-accuracy: " << std::to_string(prompt_loose_accuracy) << std::endl;
  std::cout << "Prompt-level strict-accuracy: " << std::to_string(prompt_strict_accuracy) << std::endl;

  std::cout << "Details:" << std::endl;

  for (auto sample_id : used_sample_ids_) {
    std::cout << "{\"key\": "<< std::to_string(samples_[sample_id]->key) << ", ";
    bool follow_all = true;
    bool first = true;
    std::cout << "\"follow_instruction_list_loose\": [";
    for (bool item : samples_[sample_id]->status_loose) {
        if (!first) std::cout << ", ";
        first = false;
        if (!item) follow_all = false;
        std::cout << (item ? "true" : "false");
    }
    std::cout << "], " << "\"follow_all_instructions_loose\": " << (follow_all ? "true" : "false") << ", ";

    follow_all = true;
    first = true;
    std::cout << "\"follow_instruction_list_strict\": [";
    for (bool item : samples_[sample_id]->status_strict) {
        if (!first) std::cout << ", ";
        first = false;
        if (!item) follow_all = false;
        std::cout << (item ? "true" : "false");
    }
    std::cout << "], " << "\"follow_all_instructions_strict\": " << (follow_all ? "true" : "false") << "}" << std::endl;
  }


  return (instruction_loose_accuracy + instruction_strict_accuracy +
          prompt_loose_accuracy + prompt_strict_accuracy) /
         4.0f;
}

std::string IFEval::ComputeAccuracyString() {
  float acc = ComputeAccuracy();
  return "Accuracy: " + std::to_string(acc * 100.0f) + "%";
}

inline std::vector<std::unique_ptr<ifeval::Instruction>>
IFEval::BuildInstructions(const crow::json::rvalue& ex) {
  std::vector<std::unique_ptr<ifeval::Instruction>> out;

  // ---- helpers (local) ----
  auto parse_relation = [](const std::string& s) -> ifeval::Relation {
    return (s == "at least") ? ifeval::Relation::AT_LEAST
                             : ifeval::Relation::LESS_THAN;
  };

  auto add = [&](auto ptr) { out.emplace_back(std::move(ptr)); };

  auto get_strs = [&](const std::string& key, int i) -> std::vector<std::string> {
    std::vector<std::string> svals;
    auto vals = ex["kwargs"][i][key].as_list();

    for (auto val : vals) {
        svals.emplace_back(val.as_string());
    }

    return svals;
  };
  auto get_str = [&](const std::string& key, int i) -> std::string {
      return ex["kwargs"][i][key].as_string();
  };
  auto get_int = [&](const std::string& key, int i) -> int64_t {
      return ex["kwargs"][i][key].as_i();
  };

  std::vector<crow::json::rvalue> ids(ex["instruction_id_list"].as_list());
  if (ids.empty()) return out;

  // Enum for switch (one case per instruction kind)
  enum class Kind {
    kCapitalWordFrequency,
    kEnglishCapital,
    kEnglishLowercase,
    kRepeatPrompt,
    kTwoResponses,
    kNumberPlaceholders,
    kPostscript,
    kConstrainedResponse,
    kJsonFormat,
    kMultipleSections,
    kNumberBulletLists,
    kNumberHighlightedSections,
    kTitle,
    kExistence,
    kForbiddenWords,
    kFrequency,
    kLetterFrequency,
    kResponseLanguage,
    kNthParagraphFirstWord,
    kNumberParagraphs,
    kNumberSentences,
    kNumberWords,
    kNoComma,
    kEndChecker,
    kQuotation,
    kUnknown
  };

  auto to_kind = [](const std::string& id) -> Kind {
    auto colon = id.find(':');
    std::string name = (colon == std::string::npos) ? id : id.substr(colon + 1);
    if (name == "capital_word_frequency") return Kind::kCapitalWordFrequency;
    if (name == "english_capital") return Kind::kEnglishCapital;
    if (name == "english_lowercase") return Kind::kEnglishLowercase;
    if (name == "repeat_prompt") return Kind::kRepeatPrompt;
    if (name == "two_responses") return Kind::kTwoResponses;
    if (name == "number_placeholders") return Kind::kNumberPlaceholders;
    if (name == "postscript") return Kind::kPostscript;
    if (name == "constrained_response") return Kind::kConstrainedResponse;
    if (name == "json_format") return Kind::kJsonFormat;
    if (name == "multiple_sections") return Kind::kMultipleSections;
    if (name == "number_bullet_lists") return Kind::kNumberBulletLists;
    if (name == "number_highlighted_sections") return Kind::kNumberHighlightedSections;
    if (name == "title") return Kind::kTitle;
    if (name == "existence") return Kind::kExistence;
    if (name == "forbidden_words") return Kind::kForbiddenWords;
    if (name == "frequency") return Kind::kFrequency;
    if (name == "letter_frequency") return Kind::kLetterFrequency;
    if (name == "response_language") return Kind::kResponseLanguage;
    if (name == "nth_paragraph_first_word") return Kind::kNthParagraphFirstWord;
    if (name == "number_paragraphs") return Kind::kNumberParagraphs;
    if (name == "number_sentences") return Kind::kNumberSentences;
    if (name == "number_words") return Kind::kNumberWords;
    if (name == "no_comma") return Kind::kNoComma;
    if (name == "end_checker") return Kind::kEndChecker;
    if (name == "quotation") return Kind::kQuotation;
    return Kind::kUnknown;
  };

  // Build each instruction from kwargs/<i>/* using
  // tensorflow::GetFeatureValues(ex, key, &vec)
  for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
    const std::string id = ids[i].as_string();
    const Kind kind = to_kind(id);

    switch (kind) {
      case Kind::kCapitalWordFrequency: {
        int pct = get_int("capital_frequency", i);
        std::string rel = get_str("capital_relation", i);
        add(std::make_unique<ifeval::CapitalWordFrequency>(pct, parse_relation(rel)));
        break;
      }
      case Kind::kEnglishCapital: {
        add(std::make_unique<ifeval::EnglishCapital>());
        break;
      }
      case Kind::kEnglishLowercase: {
        add(std::make_unique<ifeval::EnglishLowercase>());
        break;
      }
      case Kind::kRepeatPrompt: {
        std::string p = get_str("prompt_to_repeat", i);
        add(std::make_unique<ifeval::RepeatPrompt>(p));
        break;
      }
      case Kind::kTwoResponses: {
        add(std::make_unique<ifeval::TwoResponses>());
        break;
      }
      case Kind::kNumberPlaceholders: {
        int n = get_int("num_placeholders", i);
        add(std::make_unique<ifeval::NumberPlaceholders>(n));
        break;
      }
      case Kind::kPostscript: {
        std::string m = get_str("postscript_marker", i);
        add(std::make_unique<ifeval::Postscript>(m));
        break;
      }
      case Kind::kConstrainedResponse: {
        add(std::make_unique<ifeval::ConstrainedResponse>());
        break;
      }
      case Kind::kJsonFormat: {
        add(std::make_unique<ifeval::JsonFormat>());
        break;
      }
      case Kind::kMultipleSections: {
        int n = get_int("num_sections", i);
        std::string sep = get_str("section_spliter", i);
        add(std::make_unique<ifeval::MultipleSections>(n, sep));
        break;
      }
      case Kind::kNumberBulletLists: {
        int n = get_int("num_bullets", i);
        add(std::make_unique<ifeval::NumberBulletLists>(n));
        break;
      }
      case Kind::kNumberHighlightedSections: {
        int n = get_int("num_highlights", i);
        add(std::make_unique<ifeval::NumberHighlightedSections>(n));
        break;
      }
      case Kind::kTitle: {
        add(std::make_unique<ifeval::Title>());
        break;
      }
      case Kind::kExistence: {
        std::vector<std::string> kws = get_strs("keywords", i);
        add(std::make_unique<ifeval::Existence>(kws));
        break;
      }
      case Kind::kForbiddenWords: {
        std::vector<std::string> bad = get_strs("forbidden_words", i);
        add(std::make_unique<ifeval::ForbiddenWords>(bad));
        break;
      }
      case Kind::kFrequency: {
        int n = get_int("frequency", i);
        std::string kw = get_str("keyword", i);
        std::string rel = get_str("relation", i);
        add(std::make_unique<ifeval::Frequency>(n, kw, parse_relation(rel)));
        break;
      }
      case Kind::kLetterFrequency: {
        int n = get_int("let_frequency", i);
        std::string letter = get_str("letter", i);
        std::string rel = get_str("let_relation", i);
        char ch = letter.empty() ? 'a' : letter[0];
        add(std::make_unique<ifeval::LetterFrequency>(n, ch, parse_relation(rel)));
        break;
      }
      case Kind::kResponseLanguage: {
        std::string lang = get_str("language", i);
        add(std::make_unique<ifeval::ResponseLanguage>(lang));
        break;
      }
      case Kind::kNthParagraphFirstWord: {
        int nth = get_int("nth_paragraph", i);
        int total = get_int("num_paragraphs", i);
        std::string fw = get_str("first_word", i);
        add(std::make_unique<ifeval::NthParagraphFirstWord>(nth, fw, total));
        break;
      }
      case Kind::kNumberParagraphs: {
        int n = get_int("num_paragraphs", i);
        add(std::make_unique<ifeval::NumberParagraphs>(n));
        break;
      }
      case Kind::kNumberSentences: {
        int n = get_int("num_sentences", i);
        std::string rel = get_str("relation", i);
        add(std::make_unique<ifeval::NumberSentences>(n, parse_relation(rel)));
        break;
      }
      case Kind::kNumberWords: {
        int n = get_int("num_words", i);
        std::string rel = get_str("relation", i);
        add(std::make_unique<ifeval::NumberWords>(n, parse_relation(rel)));
        break;
      }
      case Kind::kNoComma: {
        add(std::make_unique<ifeval::NoComma>());
        break;
      }
      case Kind::kEndChecker: {
        std::string end = get_str("end_phrase", i);
        add(std::make_unique<ifeval::EndChecker>(end));
        break;
      }
      case Kind::kQuotation: {
        add(std::make_unique<ifeval::Quotation>());
        break;
      }
      case Kind::kUnknown:
      default: {
        // Unknown instruction id: skip (or handle as needed)
        break;
      }
    }
  }

  return out;
}

}  // namespace mobile
}  // namespace mlperf
