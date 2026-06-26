# Rakyat Speaks: Public Sentiment on Malaysian Government Policies

An NLP application that classifies Malaysian public sentiment toward government
policies as positive, negative, or neutral. Built for SAIA 2163 Natural Language
Processing, Final Project.

## Problem Statement

Government policies such as fuel subsidy rationalisation, taxation, and cost-of-living
measures affect millions of Malaysians, yet measuring genuine public sentiment at scale
is difficult. Traditional surveys are slow, costly, and limited in reach. Social media
discourse offers a large, organic, real-time signal of public opinion, but it is
unstructured and written in mixed Malay-English (Manglish), making it hard to analyse
manually.

This project builds an NLP system that automatically classifies public sentiment on
Malaysian government policies from real social media discussion, turning thousands of
unstructured comments into measurable insight.

## Objectives

1. Build a labeled dataset of Malaysian public opinion on government policies.
2. Develop an NLP pipeline that classifies sentiment as positive, negative, or neutral.
3. Train and compare multiple machine learning models for sentiment classification.
4. Deliver an interactive web application where users can input text and receive
   instant sentiment predictions.
5. Produce visual insights into how Malaysians feel about key policy areas.

## Dataset

The dataset consists of real public comments and discussion threads sourced from the
r/Malaysia subreddit, covering six policy areas: fuel subsidy, cost of living, tax,
economy, general government, and social policy.

- 7,383 labeled text samples
- Three sentiment classes: positive, negative, neutral
- Mixed Malay-English (Manglish) content reflecting authentic Malaysian discourse
- Stored in data/malaysian_sentiment_labeled.csv

## Project Scope

The project covers the full NLP workflow:

- Data collection and preparation
- Text preprocessing and feature extraction
- Model training, comparison, and evaluation
- An interactive Streamlit web application
- Visual dashboards and reporting

## Repository Structure

- data/ : datasets
- notebooks/ : model development notebooks
- models/ : saved models
- app.py : Streamlit application

## Project Status

As of 26 June 2026, the dataset is complete and ready for the modelling phase.

## Next Steps

- NLP pipeline: text preprocessing, feature extraction (TF-IDF and Bag of Words),
  and model training (Naive Bayes and SVM).
- Streamlit application: five-page interface for predictions, data exploration,
  and visual insights.
- Visualisations: word clouds, class distribution, model comparison, and confusion matrix.
- Documentation: technical report, presentation slides, and project poster.

## Team

SAIA 2163 Group Project.