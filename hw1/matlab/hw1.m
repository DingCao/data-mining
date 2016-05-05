% hw1.m: the main-flow of the trainning.
% COPY RIGHTS (C) HUANGJUNJIE@SYSU(SNO: 13331087). ALL RIGHTS RESESRVED.
 
clear;
close all; 
clc

% initial variables
folds = 5;
n = 384;
lambda_vec = [0.001; 0.003; 0.01; 0.03; 0.1; 0.3; 1; 3; 10];
%lambda_vec = [0.1; 0.3; 1; 3; 10];
num_iters = 10000;
alpha = 0.03;

%% ======================== preprocessing data ===========================
% Loads the data. Remembers that there are fields in the first line
% and is an id in the first columns
 
loadData;

%% ========================= Choose Lambda ================================

%plotCrossCurve

%% ========================= Learning Curve ===============================

%plotlearningCurve

%% ========================= Prediction ===============================

predict