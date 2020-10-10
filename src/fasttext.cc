/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fasttext.h"
#include "loss.h"
#include "quantmatrix.h"
#include "stdlib.h"

#include <algorithm>
#include <bits/stdint-intn.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <random>

namespace fasttext
{

  constexpr int32_t FASTTEXT_VERSION = 12; /* Version 1b */
  constexpr int32_t FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

  bool comparePairs(
      const std::pair<real, std::string> &l,
      const std::pair<real, std::string> &r);

  std::shared_ptr<Loss> FastText::createLoss(std::shared_ptr<Matrix> &output)
  {
    loss_name lossName = args_->loss;
    switch (lossName)
    {
    case loss_name::hs:
      return std::make_shared<HierarchicalSoftmaxLoss>(
          output, getTargetCounts());
    case loss_name::ns:
      return std::make_shared<NegativeSamplingLoss>(
          output, args_->neg, getTargetCounts());
    case loss_name::softmax:
      return std::make_shared<SoftmaxLoss>(output);
    case loss_name::ova:
      return std::make_shared<OneVsAllLoss>(output);
    default:
      throw std::runtime_error("Unknown loss");
    }
  }

  FastText::FastText()
      : quant_(false), wordVectors_(nullptr), trainException_(nullptr) {}

  void FastText::addInputVector(Vector &vec, int32_t ind) const
  {
    vec.addRow(*input_, ind);
  }

  std::shared_ptr<const Dictionary> FastText::getDictionary() const
  {
    return dict_;
  }

  const Args FastText::getArgs() const
  {
    return *args_.get();
  }

  std::shared_ptr<const DenseMatrix> FastText::getInputMatrix() const
  {
    if (quant_)
    {
      throw std::runtime_error("Can't export quantized matrix");
    }
    assert(input_.get());
    return std::dynamic_pointer_cast<DenseMatrix>(input_);
  }

  void FastText::setMatrices(
      const std::shared_ptr<DenseMatrix> &inputMatrix,
      const std::shared_ptr<DenseMatrix> &outputMatrix)
  {
    assert(input_->size(1) == output_->size(1));

    input_ = std::dynamic_pointer_cast<Matrix>(inputMatrix);
    output_ = std::dynamic_pointer_cast<Matrix>(outputMatrix);
    wordVectors_.reset();
    args_->dim = input_->size(1);

    buildModel();
  }

  std::shared_ptr<const DenseMatrix> FastText::getOutputMatrix() const
  {
    if (quant_ && args_->qout)
    {
      throw std::runtime_error("Can't export quantized matrix");
    }
    assert(output_.get());
    return std::dynamic_pointer_cast<DenseMatrix>(output_);
  }

  int32_t FastText::getWordId(const std::string &word) const
  {
    return dict_->getId(word);
  }

  int32_t FastText::getSubwordId(const std::string &subword) const
  {
    int32_t h = dict_->hash(subword) % args_->bucket;
    return dict_->nwords() + h;
  }

  void FastText::getWordVector(Vector &vec, const std::string &word) const
  {
    const std::vector<int32_t> &ngrams = dict_->getSubwords(word);
    vec.zero();
    for (int i = 0; i < ngrams.size(); i++)
    {
      addInputVector(vec, ngrams[i]);
    }
    if (ngrams.size() > 0)
    {
      vec.mul(1.0 / ngrams.size());
    }
  }

  void FastText::getSubwordVector(Vector &vec, const std::string &subword) const
  {
    vec.zero();
    int32_t h = dict_->hash(subword) % args_->bucket;
    h = h + dict_->nwords();
    addInputVector(vec, h);
  }

  void FastText::saveVectors(const std::string &filename)
  {
    if (!input_ || !output_)
    {
      throw std::runtime_error("Model never trained");
    }
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
      throw std::invalid_argument(
          filename + " cannot be opened for saving vectors!");
    }
    ofs << dict_->nwords() << " " << args_->dim << std::endl;
    Vector vec(args_->dim);
    for (int32_t i = 0; i < dict_->nwords(); i++)
    {
      std::string word = dict_->getWord(i);
      getWordVector(vec, word);
      ofs << word << " " << vec << std::endl;
    }
    ofs.close();
  }

  void FastText::saveOutput(const std::string &filename)
  {
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
      throw std::invalid_argument(
          filename + " cannot be opened for saving vectors!");
    }
    if (quant_)
    {
      throw std::invalid_argument(
          "Option -saveOutput is not supported for quantized models.");
    }
    int32_t n =
        (args_->model == model_name::sup) ? dict_->nlabels() : dict_->nwords();
    ofs << n << " " << args_->dim << std::endl;
    Vector vec(args_->dim);
    for (int32_t i = 0; i < n; i++)
    {
      std::string word = (args_->model == model_name::sup) ? dict_->getLabel(i)
                                                           : dict_->getWord(i);
      vec.zero();
      vec.addRow(*output_, i);
      ofs << word << " " << vec << std::endl;
    }
    ofs.close();
  }

  bool FastText::checkModel(std::istream &in)
  {
    int32_t magic;
    in.read((char *)&(magic), sizeof(int32_t));
    if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32)
    {
      return false;
    }
    in.read((char *)&(version), sizeof(int32_t));
    if (version > FASTTEXT_VERSION)
    {
      return false;
    }
    return true;
  }

  void FastText::signModel(std::ostream &out)
  {
    const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
    const int32_t version = FASTTEXT_VERSION;
    out.write((char *)&(magic), sizeof(int32_t));
    out.write((char *)&(version), sizeof(int32_t));
  }

  void FastText::saveModel(const std::string &filename)
  {
    std::ofstream ofs(filename, std::ofstream::binary);
    if (!ofs.is_open())
    {
      throw std::invalid_argument(filename + " cannot be opened for saving!");
    }
    if (!input_ || !output_)
    {
      throw std::runtime_error("Model never trained");
    }
    signModel(ofs);
    args_->save(ofs);
    dict_->save(ofs);

    ofs.write((char *)&(quant_), sizeof(bool));
    input_->save(ofs);

    ofs.write((char *)&(args_->qout), sizeof(bool));
    output_->save(ofs);

    ofs.close();
  }

  void FastText::loadModel(const std::string &filename)
  {
    std::ifstream ifs(filename, std::ifstream::binary);
    if (!ifs.is_open())
    {
      throw std::invalid_argument(filename + " cannot be opened for loading!");
    }
    if (!checkModel(ifs))
    {
      throw std::invalid_argument(filename + " has wrong file format!");
    }
    loadModel(ifs);
    ifs.close();
  }

  std::vector<int64_t> FastText::getTargetCounts() const
  {
    if (args_->model == model_name::sup)
    {
      return dict_->getCounts(entry_type::label);
    }
    else
    {
      return dict_->getCounts(entry_type::word);
    }
  }

  void FastText::buildModel()
  {
    auto loss = createLoss(output_);
    bool normalizeGradient = (args_->model == model_name::sup);
    model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
  }

  void FastText::loadModel(std::istream &in)
  {
    args_ = std::make_shared<Args>();
    input_ = std::make_shared<DenseMatrix>();
    output_ = std::make_shared<DenseMatrix>();
    args_->load(in);
    if (version == 11 && args_->model == model_name::sup)
    {
      // backward compatibility: old supervised models do not use char ngrams.
      args_->maxn = 0;
    }
    dict_ = std::make_shared<Dictionary>(args_, in);

    bool quant_input;
    in.read((char *)&quant_input, sizeof(bool));
    if (quant_input)
    {
      quant_ = true;
      input_ = std::make_shared<QuantMatrix>();
    }
    input_->load(in);

    if (!quant_input && dict_->isPruned())
    {
      throw std::invalid_argument(
          "Invalid model file.\n"
          "Please download the updated model from www.fasttext.cc.\n"
          "See issue #332 on Github for more information.\n");
    }

    in.read((char *)&args_->qout, sizeof(bool));
    if (quant_ && args_->qout)
    {
      output_ = std::make_shared<QuantMatrix>();
    }
    output_->load(in);

    buildModel();
  }

  std::tuple<int64_t, double, double> FastText::progressInfo(real progress)
  {
    double t = utils::getDuration(start_, std::chrono::steady_clock::now());
    double lr = args_->lr * (1.0 - progress);
    double wst = 0;

    int64_t eta = 2592000; // Default to one month in seconds (720 * 3600)

    if (progress > 0 && t >= 0)
    {
      eta = t * (1 - progress) / progress;
      wst = double(tokenCount_) / t / args_->thread;
    }

    return std::tuple<double, double, int64_t>(wst, lr, eta);
  }

  void FastText::printInfo(real progress, real loss, std::ostream &log_stream)
  {
    double wst;
    double lr;
    int64_t eta;
    std::tie<double, double, int64_t>(wst, lr, eta) = progressInfo(progress);

    log_stream << std::fixed;
    log_stream << "Progress: ";
    log_stream << std::setprecision(1) << std::setw(5) << (progress * 100) << "%";
    log_stream << " words/sec/thread: " << std::setw(7) << int64_t(wst);
    log_stream << " lr: " << std::setw(9) << std::setprecision(6) << lr;
    log_stream << " avg.loss: " << std::setw(9) << std::setprecision(6) << loss;
    log_stream << " ETA: " << utils::ClockPrint(eta);
    log_stream << std::flush;
  }

  std::vector<int32_t> FastText::selectEmbeddings(int32_t cutoff) const
  {
    std::shared_ptr<DenseMatrix> input =
        std::dynamic_pointer_cast<DenseMatrix>(input_);
    Vector norms(input->size(0));
    input->l2NormRow(norms);
    std::vector<int32_t> idx(input->size(0), 0);
    std::iota(idx.begin(), idx.end(), 0);
    auto eosid = dict_->getId(Dictionary::EOS);
    std::sort(idx.begin(), idx.end(), [&norms, eosid](size_t i1, size_t i2) {
      if (i1 == eosid && i2 == eosid)
      { // satisfy strict weak ordering
        return false;
      }
      return eosid == i1 || (eosid != i2 && norms[i1] > norms[i2]);
    });
    idx.erase(idx.begin() + cutoff, idx.end());
    return idx;
  }

  void FastText::quantize(const Args &qargs, const TrainCallback &callback)
  {
    if (args_->model != model_name::sup)
    {
      throw std::invalid_argument(
          "For now we only support quantization of supervised models");
    }
    args_->input = qargs.input;
    args_->qout = qargs.qout;
    args_->output = qargs.output;
    std::shared_ptr<DenseMatrix> input =
        std::dynamic_pointer_cast<DenseMatrix>(input_);
    std::shared_ptr<DenseMatrix> output =
        std::dynamic_pointer_cast<DenseMatrix>(output_);
    bool normalizeGradient = (args_->model == model_name::sup);

    if (qargs.cutoff > 0 && qargs.cutoff < input->size(0))
    {
      auto idx = selectEmbeddings(qargs.cutoff);
      dict_->prune(idx);
      std::shared_ptr<DenseMatrix> ninput =
          std::make_shared<DenseMatrix>(idx.size(), args_->dim);
      for (auto i = 0; i < idx.size(); i++)
      {
        for (auto j = 0; j < args_->dim; j++)
        {
          ninput->at(i, j) = input->at(idx[i], j);
        }
      }
      input = ninput;
      if (qargs.retrain)
      {
        args_->epoch = qargs.epoch;
        args_->lr = qargs.lr;
        args_->thread = qargs.thread;
        args_->verbose = qargs.verbose;
        auto loss = createLoss(output_);
        model_ = std::make_shared<Model>(input, output, loss, normalizeGradient);
        startThreads(callback);
      }
    }
    input_ = std::make_shared<QuantMatrix>(
        std::move(*(input.get())), qargs.dsub, qargs.qnorm);

    if (args_->qout)
    {
      output_ = std::make_shared<QuantMatrix>(
          std::move(*(output.get())), 2, qargs.qnorm);
    }
    quant_ = true;
    auto loss = createLoss(output_);
    model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
  }

  void FastText::supervised(
      Model::State &state,
      real lr,
      const std::vector<int32_t> &line,
      const std::vector<int32_t> &labels)
  {
    if (labels.size() == 0 || line.size() == 0)
    {
      return;
    }
    if (args_->loss == loss_name::ova)
    {
      model_->update(line, labels, Model::kAllLabelsAsTarget, lr, state);
    }
    else
    {
      std::uniform_int_distribution<> uniform(0, labels.size() - 1);
      int32_t i = uniform(state.rng);
      model_->update(line, labels, i, lr, state);
    }
  }

  void FastText::cbow(
      Model::State &state,
      real lr,
      const std::vector<int32_t> &line)
  {
    std::vector<int32_t> bow;
    std::uniform_int_distribution<> uniform(1, args_->ws);
    for (int32_t w = 0; w < line.size(); w++)
    {
      int32_t boundary = uniform(state.rng);
      bow.clear();
      for (int32_t c = -boundary; c <= boundary; c++)
      {
        if (c != 0 && w + c >= 0 && w + c < line.size())
        {
          const std::vector<int32_t> &ngrams = dict_->getSubwords(line[w + c]);
          bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
        }
      }
      model_->update(bow, line, w, lr, state);
    }
  }

  void FastText::skipgram(
      Model::State &state,
      real lr,
      const std::vector<int32_t> &line)
  {
    std::uniform_int_distribution<> uniform(1, args_->ws);
    for (int32_t w = 0; w < line.size(); w++)
    {
      int32_t boundary = uniform(state.rng);
      const std::vector<int32_t> &ngrams = dict_->getSubwords(line[w]);
      for (int32_t c = -boundary; c <= boundary; c++)
      {
        if (c != 0 && w + c >= 0 && w + c < line.size())
        {
          model_->update(ngrams, line, w + c, lr, state);
        }
      }
    }
  }


  void FastText::skipgramDistill(
                                 Model::State &state,
                                 Model::State &big_state,
                                 real lr,
                                 const std::vector<int32_t> &line,
                                 std::vector<std::pair<int32_t, std::shared_ptr<Vector>>> &temp_nn_vectors,
                                 Vector &temp_out_vector)
  {
    std::uniform_int_distribution<> uniform(1, args_->ws);
    for (int32_t w = 0; w < line.size(); w++)
      {
      int32_t boundary = uniform(state.rng);
      const std::vector<int32_t> &ngrams = dict_->getSubwords(line[w]);
      for (int32_t c = -boundary; c <= boundary; c++)
        {
        if (c != 0 && w + c >= 0 && w + c < line.size())
          {
            big_fasttext->model_->computeHidden(ngrams, big_state);
            //std::cout << "#\nHidden head:" << big_state.hidden[0] << big_state.hidden[1] << big_state.hidden[2] << "\n";

            int32_t word_id = line[w + c];

            for(int i=0; i < NN_SIZE; i++) {
              int32_t neighbor_id = (*(big_fasttext->computed_nn))[word_id][i];
              temp_nn_vectors[i].first = neighbor_id;
              big_fasttext->getWordVector(*(temp_nn_vectors[i].second), big_fasttext->dict_->getWord(neighbor_id));
            }
            //std::cout << "Neighbor vector:" << (*temp_nn_vectors[0].second)[0] << (*temp_nn_vectors[0].second)[1] << "\n";

            big_fasttext->model_->loss_->computeOutputFast(big_state, temp_nn_vectors);
            
            //std::cout << "Output head:" << big_state.output[0] << big_state.output[1] << big_state.output[2] << big_state.output[3] << "\n";

            model_->updateDistill(ngrams, line, w + c, big_state.output, lr, state);
        }
      }
    }
  }


    void FastText::precomputeNN() {
      std::cout << "(#) NN precomputing has begun. This my take time.\n";

      using namespace std::chrono;
      steady_clock::time_point begin = steady_clock::now();
      std::cout << std::setprecision(3);

      std::vector<std::array<int32_t, NN_SIZE>> nn_list;
      std::array<int32_t, NN_SIZE> temp_array;
      Vector temp_vector(args_->dim);

      for (int32_t i = 0; i < dict_->nwords(); i++) { 
        std::string word = dict_->getWord(i);
        getWordVector(temp_vector, word);
            
        auto nn_result = getNN(*wordVectors_, temp_vector, NN_SIZE, {word});

        for(int32_t k = 0; k < NN_SIZE; k++) {
          temp_array[k] = getWordId(nn_result[k].second);
        }

        nn_list.push_back(temp_array);

        if (i % 100 == 1) {
          int64_t elapsed_time = duration_cast<seconds>(steady_clock::now() - begin).count();
          int64_t remaining_time = (int64_t) (((dict_->nwords() - i) / i) * elapsed_time);

          std::cout << "(#) " << i << " / " << dict_->nwords() 
                    << " [" << (double)(i/dict_->nwords())*100 << "% completed] [Elapsed time: "
                    << (double)elapsed_time/60 << " min] [Remaining time: " 
                    << (double)remaining_time/60 << " min]\r";
        }
      }

      computed_nn = std::make_shared<std::vector<std::array<int32_t, NN_SIZE>>>(nn_list);
      std::cout << "(#) NN precomputing completed.\n";
    }

    std::tuple<int64_t, double, double> FastText::test(
        std::istream & in, int32_t k, real threshold) {
      Meter meter;
      test(in, k, threshold, meter);

      return std::tuple<int64_t, double, double>(
          meter.nexamples(), meter.precision(), meter.recall());
    }

    void FastText::getNNFromFile(std::string &filename) {
      std::ifstream in(filename);
      std::string word;
      std::vector<std::array<int32_t, NN_SIZE>> nn_arrays;
      std::array<int32_t, NN_SIZE> temp_array;

      if(!in.good()) {
        throw std::runtime_error(filename + " does not exist.");
      }

      for(int i = 0; i < dict_->nwords(); i++) {
        for(int k = 0; k < NN_SIZE; k++) {
          in >> word;
          temp_array[k] = std::stoi(word);

          if(temp_array[k] >= dict_->nwords()) {
            std::cout << "Mistaken numbers in the NN file :" << temp_array[k] << "\n";
          }
        }
        nn_arrays.push_back(temp_array);

        if (in.eof() && i != (dict_->nwords() - 1)) {
          throw std::runtime_error("computedNN file has " + std::to_string(i) 
                                   + " lines. Expected:" + std::to_string(dict_->nwords()));
        }
      }

      computed_nn = std::make_shared<std::vector<std::array<int32_t, NN_SIZE>>> (nn_arrays);
    }

    void FastText::saveNN(std::string &filename) {
      std::ofstream out(filename);

      for(int i = 0; i < dict_->nwords(); i++) {
        for(int k = 0; k < NN_SIZE; k++) {
          std::string s = std::to_string((*computed_nn)[i][k]);
          if (!s.empty() && s[s.length()-1] == '\n') {
            s.erase(s.length()-1);
          }
          out << s << " ";
        }
        out << "\n";
      }
      out.close();

      std::cout << "NN file created to " << filename << "\n";
    }

    void FastText::test(std::istream & in, int32_t k, real threshold,
                        Meter & meter) const {
      std::vector<int32_t> line;
      std::vector<int32_t> labels;
      Predictions predictions;
      Model::State state(args_->dim, dict_->nlabels(), 0);
      in.clear();
      in.seekg(0, std::ios_base::beg);

      while (in.peek() != EOF) {
        line.clear();
        labels.clear();
        dict_->getLine(in, line, labels);

        if (!labels.empty() && !line.empty()) {
          predictions.clear();
          predict(k, line, predictions, threshold);
          meter.log(labels, predictions);
        }
      }
    }

    void FastText::predict(int32_t k, const std::vector<int32_t> &words,
                           Predictions &predictions, real threshold) const {
      if (words.empty()) {
        return;
      }
      Model::State state(args_->dim, dict_->nlabels(), 0);
      if (args_->model != model_name::sup) {
        throw std::invalid_argument(
            "Model needs to be supervised for prediction!");
      }
      model_->predict(words, k, threshold, predictions, state);
    }

    bool FastText::predictLine(std::istream & in,
                               std::vector<std::pair<real, std::string>> &
                                   predictions,
                               int32_t k, real threshold) const {
      predictions.clear();
      if (in.peek() == EOF) {
        return false;
      }

      std::vector<int32_t> words, labels;
      dict_->getLine(in, words, labels);
      Predictions linePredictions;
      predict(k, words, linePredictions, threshold);
      for (const auto &p : linePredictions) {
        predictions.push_back(
            std::make_pair(std::exp(p.first), dict_->getLabel(p.second)));
      }

      return true;
    }

    void FastText::getSentenceVector(std::istream & in,
                                     fasttext::Vector & svec) {
      svec.zero();
      if (args_->model == model_name::sup) {
        std::vector<int32_t> line, labels;
        dict_->getLine(in, line, labels);
        for (int32_t i = 0; i < line.size(); i++) {
          addInputVector(svec, line[i]);
        }
        if (!line.empty()) {
          svec.mul(1.0 / line.size());
        }
      } else {
        Vector vec(args_->dim);
        std::string sentence;
        std::getline(in, sentence);
        std::istringstream iss(sentence);
        std::string word;
        int32_t count = 0;
        while (iss >> word) {
          getWordVector(vec, word);
          real norm = vec.norm();
          if (norm > 0) {
            vec.mul(1.0 / norm);
            svec.addVector(vec);
            count++;
          }
        }
        if (count > 0) {
          svec.mul(1.0 / count);
        }
      }
    }

    std::vector<std::pair<std::string, Vector>> FastText::getNgramVectors(
        const std::string &word) const {
      std::vector<std::pair<std::string, Vector>> result;
      std::vector<int32_t> ngrams;
      std::vector<std::string> substrings;
      dict_->getSubwords(word, ngrams, substrings);
      assert(ngrams.size() <= substrings.size());
      for (int32_t i = 0; i < ngrams.size(); i++) {
        Vector vec(args_->dim);
        if (ngrams[i] >= 0) {
          vec.addRow(*input_, ngrams[i]);
        }
        result.push_back(std::make_pair(substrings[i], std::move(vec)));
      }
      return result;
    }

    void FastText::precomputeWordVectors(DenseMatrix & wordVectors) {
      Vector vec(args_->dim);
      wordVectors.zero();
      for (int32_t i = 0; i < dict_->nwords(); i++) {
        std::string word = dict_->getWord(i);
        getWordVector(vec, word);
        real norm = vec.norm();
        if (norm > 0) {
          wordVectors.addVectorToRow(vec, i, 1.0 / norm);
        }
      }
    }

    void FastText::lazyComputeWordVectors() {
      if (!wordVectors_) {
        wordVectors_ = std::unique_ptr<DenseMatrix>(
            new DenseMatrix(dict_->nwords(), args_->dim));
        precomputeWordVectors(*wordVectors_);
      }
    }

    std::vector<std::pair<real, std::string>> FastText::getNN(
        const std::string &word, int32_t k) {
      Vector query(args_->dim);

      getWordVector(query, word);

      lazyComputeWordVectors();
      assert(wordVectors_);
      return getNN(*wordVectors_, query, k, {word});
    }


    std::vector<std::pair<real, std::string>> FastText::getNN(
        const DenseMatrix &wordVectors, const Vector &query, int32_t k,
        const std::set<std::string> &banSet) {
      std::vector<std::pair<real, std::string>> heap;

      real queryNorm = query.norm();
      if (std::abs(queryNorm) < 1e-8) {
        queryNorm = 1;
      }

      for (int32_t i = 0; i < dict_->nwords(); i++) {
        std::string word = dict_->getWord(i);
        if (banSet.find(word) == banSet.end()) {
          real dp = wordVectors.dotRow(query, i);
          real similarity = dp / queryNorm;

          if (heap.size() == k && similarity < heap.front().first) {
            continue;
          }
          heap.push_back(std::make_pair(similarity, word));
          std::push_heap(heap.begin(), heap.end(), comparePairs);
          if (heap.size() > k) {
            std::pop_heap(heap.begin(), heap.end(), comparePairs);
            heap.pop_back();
          }
        }
      }
      std::sort_heap(heap.begin(), heap.end(), comparePairs);

      return heap;
    }

    std::vector<std::pair<real, std::string>> FastText::getAnalogies(
        int32_t k, const std::string &wordA, const std::string &wordB,
        const std::string &wordC) {
      Vector query = Vector(args_->dim);
      query.zero();

      Vector buffer(args_->dim);
      getWordVector(buffer, wordA);
      query.addVector(buffer, 1.0 / (buffer.norm() + 1e-8));
      getWordVector(buffer, wordB);
      query.addVector(buffer, -1.0 / (buffer.norm() + 1e-8));
      getWordVector(buffer, wordC);
      query.addVector(buffer, 1.0 / (buffer.norm() + 1e-8));

      lazyComputeWordVectors();
      assert(wordVectors_);
      return getNN(*wordVectors_, query, k, {wordA, wordB, wordC});
    }

    void FastText::readFileDumpOutput(const std::string &filename) {
      std::ifstream ifs(filename);
      if (!ifs.is_open()) {
        throw std::invalid_argument(filename +
                                    " cannot be opened for training!");
      }

      Model::State state(args_->dim, output_->size(0), args_->seed);
      const int64_t ntokens = dict_->ntokens();

      std::vector<int32_t> line;
      while (keepTraining(ntokens)) {
        tokenCount_ += dict_->getLine(ifs, line, state.rng);

        std::cout << "# [";
        for (int32_t line_char : line) {
          std::cout << dict_->getWord(line_char) << ", ";
        }
        std::cout << "]\n";

        // std::cout << dict_->get
      }

    ifs.close();
  }

  bool FastText::keepTraining(const int64_t ntokens) const
  {
    return tokenCount_ < args_->epoch * ntokens && !trainException_;
  }

  void FastText::trainThread(int32_t threadId, const TrainCallback &callback)
  {
    std::ifstream ifs(args_->input);
    utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

    Model::State state(args_->dim, output_->size(0), threadId + args_->seed);
    Model::State big_state(big_fasttext->args_->dim, big_fasttext->output_->size(0), threadId + args_->seed);

    // Create temp vectors for nn vecs
    std::vector<std::pair<int32_t, std::shared_ptr<Vector>>> temp_nn_vectors;
    Vector new_vector(big_fasttext->args_->dim);
    for(int i=0; i < NN_SIZE; i++) {
      temp_nn_vectors.push_back(std::make_pair(0, std::make_shared<Vector>(new_vector)));
    }
    Vector temp_out_vector(big_fasttext->args_->dim);

    const int64_t ntokens = dict_->ntokens();
    int64_t localTokenCount = 0;
    std::vector<int32_t> line, labels;
    uint64_t callbackCounter = 0;
    try
    {
      while (keepTraining(ntokens))
      {
        real progress = real(tokenCount_) / (args_->epoch * ntokens);
        if (callback && ((callbackCounter++ % 64) == 0))
        {
          double wst;
          double lr;
          int64_t eta;
          std::tie<double, double, int64_t>(wst, lr, eta) =
              progressInfo(progress);
          callback(progress, loss_, wst, lr, eta);
        }
        real lr = args_->lr * (1.0 - progress);
        if (args_->model == model_name::sup)
        {
          localTokenCount += dict_->getLine(ifs, line, labels);
          supervised(state, lr, line, labels);
        }
        else if (args_->model == model_name::cbow)
        {
          localTokenCount += dict_->getLine(ifs, line, state.rng);
          cbow(state, lr, line);
        }
        else if (args_->model == model_name::sg)
        {
          localTokenCount += dict_->getLine(ifs, line, state.rng);
          if(args_->distillFrom.empty())
            {
              skipgram(state, lr, line);
            }
          else
            {
              skipgramDistill(state, big_state, lr, line, temp_nn_vectors, temp_out_vector);
            }
        }
        if (localTokenCount > args_->lrUpdateRate)
        {
          tokenCount_ += localTokenCount;
          localTokenCount = 0;
          if (threadId == 0 && args_->verbose > 1)
          {
            loss_ = state.getLoss();
          }
        }
      }
    }
    catch (DenseMatrix::EncounteredNaNError &)
    {
      trainException_ = std::current_exception();
    }
    if (threadId == 0)
      loss_ = state.getLoss();
    ifs.close();
  }

  std::shared_ptr<Matrix> FastText::getInputMatrixFromFile(
      const std::string &filename) const
  {
    std::ifstream in(filename);
    std::vector<std::string> words;
    std::shared_ptr<DenseMatrix> mat; // temp. matrix for pretrained vectors
    int64_t n, dim;
    if (!in.is_open())
    {
      throw std::invalid_argument(filename + " cannot be opened for loading!");
    }

    // Handle inc and dec vectors
    bool decremental = args_->decVectors ? true : false;
    bool incremental = args_->incVectors ? true : false;
    int expandMethod = args_->expandMethod;

    if (incremental && decremental)
    {
      throw std::invalid_argument(
          "incVectors and decVectors cannot be used at the same time!");
    }

    in >> n >> dim;
    if (dim != args_->dim)
    {
      throw std::invalid_argument(
          "Dimension of pretrained vectors (" + std::to_string(dim) +
          ") does not match dimension (" + std::to_string(args_->dim) + ")!");
    }

    if (incremental)
    {
      mat = std::make_shared<DenseMatrix>(n, dim * 2);
      std::cout << "@ Incremental - Expand Method: [" << expandMethod << "] ";

      if (expandMethod) {
        std::cout << "1234-1234\n";
      } else {
        std::cout << "11-22-33-44\n";
      }
    }
    else
    {
      mat = std::make_shared<DenseMatrix>(n, dim);
    }

    for (size_t i = 0; i < n; i++)
    {
      std::string word;
      in >> word;
      words.push_back(word);
      dict_->add(word);

      if (incremental)
      {
        real val;
        for (size_t j = 0; j < dim; j++)
        {
          in >> val;

          if (expandMethod)
          {
            mat->at(i, j * 2) = val;
            mat->at(i, j * 2 + 1) = val;
          }
          else
          {
            mat->at(i, j) = val;
            mat->at(i, j + dim) = val;
          }
        }
      }
      else
      {
        for (size_t j = 0; j < dim; j++)
        {
          in >> mat->at(i, j);
        }
      }
    }
    in.close();

    std::vector<int> indexes;
    if (decremental)
    {
      std::cout << "@ Decremental\n";
      int newdim = args_->decVectors;

      for (int i = 0; i < dim; i++)
      {
        indexes.push_back(i);
      }
      shuffle(indexes.begin(), indexes.end(), std::default_random_engine(time(NULL)));
      indexes.erase(indexes.begin() + newdim, indexes.end());
      std::sort(indexes.begin(), indexes.end());

      for (auto i = indexes.begin(); i != indexes.end(); i++)
      {
        std::cout << *i << ' ';
      }
      std::cout << "\n";

      dim = newdim;
      args_->dim = newdim;
    }
    else if (incremental)
    {
      dim *= 2;
      args_->dim = args_->dim * 2;
    }

    dict_->threshold(1, 0);
    dict_->init();
    std::shared_ptr<DenseMatrix> input = std::make_shared<DenseMatrix>(
        dict_->nwords() + args_->bucket, args_->dim);
    input->uniform(1.0 / args_->dim, args_->thread, args_->seed);

    for (size_t i = 0; i < n; i++)
    {
      int32_t idx = dict_->getId(words[i]);
      if (idx < 0 || idx >= dict_->nwords())
      {
        continue;
      }

      if (decremental)
      {
        for (size_t j = 0; j < dim; j++)
        {
          input->at(idx, j) = mat->at(i, indexes[j]);
        }
      }
      else
      {
        for (size_t j = 0; j < dim; j++)
        {
          input->at(idx, j) = mat->at(i, j);
        }
      }
    }

    // Print matrix
    std::cout << "Matrix got from file.";
    input->printMatrix(input->rows(), input->cols());
    return input;
  }

  std::shared_ptr<Matrix> FastText::createRandomMatrix() const
  {
    std::shared_ptr<DenseMatrix> input = std::make_shared<DenseMatrix>(
        dict_->nwords() + args_->bucket, args_->dim);
    input->uniform(1.0 / args_->dim, args_->thread, args_->seed);

    return input;
  }

  std::shared_ptr<Matrix> FastText::createTrainOutputMatrix() const
  {
    int64_t m =
        (args_->model == model_name::sup) ? dict_->nlabels() : dict_->nwords();
    std::shared_ptr<DenseMatrix> output =
        std::make_shared<DenseMatrix>(m, args_->dim);
    output->zero();

    return output;
  }

  void FastText::train(const Args &args, const TrainCallback &callback)
  {
    args_ = std::make_shared<Args>(args);
    dict_ = std::make_shared<Dictionary>(args_);
    if (args_->input == "-")
    {
      // manage expectations
      throw std::invalid_argument("Cannot use stdin for training!");
    }
    std::ifstream ifs(args_->input);
    if (!ifs.is_open())
    {
      throw std::invalid_argument(
          args_->input + " cannot be opened for training!");
    }
    dict_->readFromFile(ifs);
    ifs.close();

    if (!args_->pretrainedVectors.empty())
    {
      input_ = getInputMatrixFromFile(args_->pretrainedVectors);
    }
    else
    {
      input_ = createRandomMatrix();
    }
    output_ = createTrainOutputMatrix();
    quant_ = false;
    auto loss = createLoss(output_);
    bool normalizeGradient = (args_->model == model_name::sup);
    model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
    startThreads(callback);
  }

  void FastText::abort()
  {
    try
    {
      throw AbortError();
    }
    catch (AbortError &)
    {
      trainException_ = std::current_exception();
    }
  }

  void FastText::startThreads(const TrainCallback &callback)
  {
    start_ = std::chrono::steady_clock::now();
    tokenCount_ = 0;
    loss_ = -1;
    trainException_ = nullptr;
    std::vector<std::thread> threads;
    if (args_->thread > 1)
    {
      for (int32_t i = 0; i < args_->thread; i++)
      {
        threads.push_back(std::thread([=]() { trainThread(i, callback); }));
      }
    }
    else
    {
      // webassembly can't instantiate `std::thread`
      trainThread(0, callback);
    }
    const int64_t ntokens = dict_->ntokens();
    // Same condition as trainThread
    while (keepTraining(ntokens))
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (loss_ >= 0 && args_->verbose > 1)
      {
        real progress = real(tokenCount_) / (args_->epoch * ntokens);
        std::cerr << "\r";
        printInfo(progress, loss_, std::cerr);
      }
    }
    for (int32_t i = 0; i < threads.size(); i++)
    {
      threads[i].join();
    }
    if (trainException_)
    {
      std::exception_ptr exception = trainException_;
      trainException_ = nullptr;
      std::rethrow_exception(exception);
    }
    if (args_->verbose > 0)
    {
      std::cerr << "\r";
      printInfo(1.0, loss_, std::cerr);
      std::cerr << std::endl;
    }
  }

  int FastText::getDimension() const
  {
    return args_->dim;
  }

  bool FastText::isQuant() const
  {
    return quant_;
  }

  bool comparePairs(
      const std::pair<real, std::string> &l,
      const std::pair<real, std::string> &r)
  {
    return l.first > r.first;
  }

  } // namespace fasttext
