#pragma once
#include<iostream>


struct IO
{
	int x_digit[10];
	float* X ;
	float* Y;
	float* Y_ref;
	float* User_X;
	float cost;
};

struct Layer_f
{
	float* X;
	float *w ;			float* dw;
	float* refw;
	float *b;			float* db;
	float *Y;

};

struct Layer_h
{
	float* X;
	float *w;			float* dw;
	float* refw ;
	float *b;			float* db;
	float *Y;

};

struct Layer_l
{
	float* X;
	float *w;			float* dw;
	float* refw;
	float* b;			float* db;
	float *Y;

};

class E_Layer
{
public:
	IO io;
	Layer_f LF;
	Layer_h LH[2];
	Layer_l LL;

	void allocMemhost();

	float SqrMean(int numb);
	
	
};

class S_Layer
{
public:
	
	void Set_All_wby_zero(E_Layer& L);

	float sigmoid(float x);
	void Comp_y(Layer_f& L);
	void Comp_y(Layer_h& L);
	void Comp_y(Layer_l& L);
	

	//data should be loaded before execution
	void Comp_upto_last(E_Layer& EL);


		
};

//
//struct IO
//{
//	int x_digit[10];
//	float* X = new float[784 * 10];
//	float f[10];
//	float t_f[10];
//};
//
//struct Layer_f
//{
//	float* X;
//	float* w = new float[784 * 20];	float* dw = new float[784 * 20];
//	float* refw = new float[784 * 20];
//	float* b = new float[20];			float* db = new float[20];
//	float* Y = new float[20];
//
//};
//
//struct Layer_h
//{
//	float* X;
//	float* w = new float[20 * 20];	float* dw = new float[20 * 20];
//	float* refw = new float[784 * 20];
//	float* b = new float[20];			float* db = new float[20];
//	float* Y = new float[20];
//
//};
//
//struct Layer_l
//{
//	float* X;
//	float* w = new float[20 * 10];		float* dw = new float[20 * 10];
//	float* refw = new float[784 * 20];
//	float* b = new float[10];			float* db = new float[10];
//	float* Y = new float[10];
//
//};