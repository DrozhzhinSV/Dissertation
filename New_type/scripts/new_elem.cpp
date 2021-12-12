#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cmath> 
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <glpk.h>
#include <algorithm>
#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


using namespace std;

const double EPS = 5.0E-2;
const double constr = 1.0E-4;


/* Функция для определения итераций, на которых будут
 * внесены изменения
 *            
 *         
 *     Вход: 
 *         count_iter     - количество итераций
 *         lambda         - вероятность внесения изменений
 *         count_new_type - максимальное количество изменений
 *      
 *     Выход:
 *         x - вектор, значения элементов которого, определяют
 *             поведение системы на каждой итерации:
 *  
 *             0 - без изменений 
 *             1 - внесение изменений                                                                                                                                         
 */


gsl_vector *poisson_new_element(int count_iter, double lambda, int count_new_type)
{
	const gsl_rng_type *T;
	gsl_rng *r;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	
	double a = 0, b = 1;
	gsl_vector *x = gsl_vector_alloc(count_iter + 1);
	gsl_vector_set(x, 0, 0);
	
	int cnt = 0; 
    for(int i = 1; i <= count_iter; i++)
    {
		if (gsl_ran_flat(r, a, b) <= lambda and cnt < count_new_type) 
		{
			cnt++;
			gsl_vector_set(x, i, 1);
		}
		else gsl_vector_set(x, i, 0);
	}
	
	gsl_rng_free(r);
	
	return x;
}



/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*
 * Функция для определения коэффициента BETA
 *                    
 *     Вход: 
 *         count_iter - кол-во итераций
 *         poisson_vector - вектор, элементы которого, определяют
 *                          поведение системы на каждой итерации
 *      
 *     Выход: 
 *         x - вектор со значениями коэффициента beta на каждой итерации:
 * 
 *             0 - если изменений не будет
 *             число от 0.5 до 0.999 - если будет добавлен новый элемент                                                                                                                                                 */
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
gsl_vector *get_beta_vector(int count_iter, gsl_vector *poisson_vector)
{
	const gsl_rng_type *T;
	gsl_rng *r;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	
	double a = 0.5, b = 0.999;
	gsl_vector *x = gsl_vector_alloc(count_iter + 1);
	
    for(int i = 0; i <= count_iter; i++)
    {
		if (gsl_vector_get(poisson_vector, i) == 1) 
			gsl_vector_set(x, i, gsl_ran_flat(r, a, b));
		else gsl_vector_set(x, i, 0);
		gsl_ran_flat(r, a, b);
	}
	
	gsl_rng_free(r);
	
	return x;
}



/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*Определяем на каких итерациях будем решать ОДУ:
                   
                    1. Если на текущем или следующем шаге будет добавляться новый вид
                    2. Если текущий шаг кратен заданному шагу решения (solve step)
                    3. Если теущий шаг первый (нулевой, т.е. когда системы исходная)                                                                                          */
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

gsl_vector *get_step_odu(int count_iter, int solve_step, gsl_vector *poisson_vector)
{
	int j = 0;
	gsl_vector *x = gsl_vector_alloc(count_iter + 1);
	gsl_vector_set(x, 0, 1);
	
	for(int i = 1; i <= count_iter; i++)
	{
		if(gsl_vector_get(poisson_vector, i) > 0)
		{
			gsl_vector_set(x, i, 1);
			gsl_vector_set(x, i - 1, 1);
		}
		else gsl_vector_set(x, i, 0); 
		
		if(j == solve_step)
		{
			gsl_vector_set(x, i, 1);
			j = -1;
		} 
		
		j++;
	}
	
	return x;
}


/* Функция для определения неподвижной точки
  *
  *     Поскольку в положении равновесия среднии приспособленности всех видов равны,
  *     то для отыскания неподвижной точки достаточно решить СЛАУ, в которой
  *     первые (sizeA - 1) уравнений - это разность средней приспособленности 1-го элемента
  *     со средними приспособленностями остальных элементов, а последнее уравнение - 
  *     это условие нормировки - сумма всех частот равна 1  
  *
  *     Вход:
  *         sizeA - размер матрицы ландшафта приспособленности
  *         A     - матрица ландшафта приспособленности
  *
  *     Выход:
  *         x     - координаты неподвижной точки
 */
gsl_vector *get_freq(int sizeA, gsl_matrix *A)
{
    gsl_matrix *left_part = gsl_matrix_alloc(sizeA, sizeA);

    gsl_vector *v1 = gsl_vector_alloc(sizeA);
    gsl_vector *v2 = gsl_vector_alloc(sizeA);

    for(int j = 1; j < sizeA; j++)
    {
        gsl_matrix_get_row(v1, A, 0);
        gsl_matrix_get_row(v2, A, j);
        gsl_vector_sub(v1, v2);
        gsl_matrix_set_row(left_part, (j - 1), v1);            
    }
    gsl_vector_set_all(v1, 1);
    gsl_matrix_set_row(left_part, (sizeA - 1), v1);
    gsl_vector_free(v1);
    gsl_vector_free(v2);

    gsl_vector *right_part = gsl_vector_calloc(sizeA);
    gsl_vector_set(right_part, (sizeA - 1), 1);

    gsl_vector *x = gsl_vector_alloc(sizeA);
    
    int s;
    gsl_permutation *p = gsl_permutation_alloc(sizeA);
    gsl_linalg_LU_decomp(left_part, p, &s);
    gsl_linalg_LU_solve(left_part, p, right_part, x);
    
    gsl_vector_free(right_part);
    gsl_matrix_free(left_part);
    gsl_permutation_free(p);

    return x;    
}




/* Функция для решения ОДУ */
int func (double t, const double y[], double f[], void *params)
{
    (void)(t); 
    gsl_matrix *A = (gsl_matrix*)params;
        
    for(int i = 0; i < A->size1; i++)
    {
		f[i] = 0;
		for(int j = 0; j < A->size1; j++)
		{
			f[i] = f[i] + gsl_matrix_get(A, i, j) * y[j];
			for(int k = 0; k < A->size1; k++)
			{
				f[i] = f[i] - gsl_matrix_get(A, k, j) * y[k] * y[j];
			}
		}
		f[i] = f[i] * y[i];
	}
    
    return GSL_SUCCESS;
}

int jac (double t, const double y[], double *dfdy, double dfdt[], void *params)
{
    (void)(t); 
    gsl_matrix *A = (gsl_matrix*)params;
    gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, A->size1, A->size1);
    gsl_matrix * m = &dfdy_mat.matrix;
    
    for(int i = 0; i < A->size1; i++)
    {
		for(int j = 0; j < A->size1; j++)
		{
			gsl_matrix_set(m, i, j, y[i] * gsl_matrix_get(A, i, j));
			for(int k = 0; k < A->size1; k++)
			    gsl_matrix_set(m, i, j, gsl_matrix_get(m, i, j) - y[i] * y[k] * (gsl_matrix_get(A, k, j) + gsl_matrix_get(A, j, k)));
			
			if (i == j)
			{
				for(int k = 0; k < A->size1; k++)
				{
					gsl_matrix_set(m, i, i, gsl_matrix_get(m, i, i) + y[k] * gsl_matrix_get(A, i, k));
				    for(int l = 0; l < A->size1; l++)
				       gsl_matrix_set(m, i, i, gsl_matrix_get(m, i, i) - y[k] * y[l] * gsl_matrix_get(A, k, l));
				} 
			} 
		}
	}
    
    for(int i = 0; i < A->size1; i++)
        dfdt[i] = 0.0;
 
    return GSL_SUCCESS;
}


/* Функция для вычисления среднего интегрального фитнеса */
double get_avg_integral_fitness(gsl_vector *U_continuos, gsl_matrix *A, int sizeA, double count_step, int sizeU_)
{
	double s, f = 0;
	int start = sizeU_ - sizeA * (count_step + 1);
	for(int i = 0; i <= count_step; i++)
	{
		s = 0;
		for(int j = 0; j < sizeA; j++)
		    for(int k = 0; k < sizeA; k++)
		        s = s + gsl_matrix_get(A, j, k) * gsl_vector_get(U_continuos, start + i * sizeA + j) * gsl_vector_get(U_continuos, start + i * sizeA + k);
		
		if ((i == 0) || (i == count_step)) s = s / 2;
		f = f + s;
    }
	   
	f = f / count_step;
	
	return f;
}


 /* Решаем задачу линейного программирования: находим приращения элементов 
  * матрицы ландшафта приспособленности
  *
  *    Вход:
  *        A - матрица ландшафта приспособленности
  *        x - координаты положения равновесия
  *        sizeA - порядок матрицы ландшафта приспособленности
  *
  *    Выход:
  *        B - матрица приращений ландшафта приспособленности
  */
gsl_matrix *solve_lin_prog(gsl_matrix *A, gsl_vector *x, int sizeA)
{
    /* Находим обратную матрицу для матрицы A */
    gsl_matrix *invA = gsl_matrix_alloc(sizeA, sizeA);
    gsl_matrix   *A2 = gsl_matrix_alloc(sizeA, sizeA);
    gsl_matrix_memcpy(A2, A);
    int s;

    gsl_permutation *p = gsl_permutation_alloc(sizeA);
    gsl_linalg_LU_decomp(A2, p, &s);
    gsl_linalg_LU_invert(A2, p, invA);
    gsl_matrix_free(A2);
    gsl_permutation_free(p);

    /* Находим коэффициенты перед приращениями */
    gsl_matrix *B = gsl_matrix_alloc(sizeA, sizeA);
    gsl_matrix_set_zero(B);
    double a;

    for(int i = 0; i < sizeA; i++)
    {
        for(int j = 0; j < sizeA; j++)
        {
            for(int k = 0; k < sizeA; k++)
            {
                a = 0;
                for(int m = 0; m < sizeA; m++)
                {
                    a = a + gsl_matrix_get(invA, j, m);
                }
                gsl_matrix_set(B, i, j, gsl_matrix_get(B, i, j) + a * gsl_matrix_get(invA, k, i));
            }
        }
    } 

    /* Ставим ЗЛП */
    glp_prob *lp;
    lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MAX);

    /* Добавляем ограничения:
     *    - на приращения (они должны быть меньше некоторой малой величины constr, заданной глобально)
     *    - на сумму произведений элементов матрицы ландшафта приспособленности и приращений
     *    - на элементы неподвижной точки: после получения новой матрицы ландшафта приспособленности
     *      они не должны попадать на границу, т.е. система не должна вырождаться
     */
    int count_chng = 0, count_chng2 = 2;
    for(int i = 0; i < sizeA; i++)
        if((gsl_vector_get(x, i) <= EPS) || (gsl_vector_get(x, i) >= (1 - EPS))) count_chng++;
    
    glp_add_rows(lp, 1 + count_chng);
    glp_set_row_bnds(lp, 1, GLP_UP, 0.0, 0.0);
    
    if(count_chng > 0)
    { 
		for(int i = 0; i < sizeA; i++)
		{
			if(gsl_vector_get(x, i) <= EPS)
			{
				glp_set_row_bnds(lp, count_chng2, GLP_LO, 0, 0);
				count_chng2++;
			}
			
			if(gsl_vector_get(x, i) >= (1 - EPS))
			{
				glp_set_row_bnds(lp, count_chng2, GLP_UP, 0, 0);
				count_chng2++;
			}
			
			if((count_chng2 - 2) >= count_chng) break;
		}
    }   

    int ia[(count_chng + 1) * sizeA * sizeA + 1], ja[(count_chng + 1) * sizeA * sizeA + 1];
    double ar[(count_chng + 1) * sizeA * sizeA + 1];
    int ind1 = 1;
    count_chng2 = 1;

    for(int k = 1; k <= (sizeA + 1); k++)
    {
		if((k == 1) || (gsl_vector_get(x, k - 2) <= EPS) || (gsl_vector_get(x, k - 2) >= (1 - EPS)))
		{
			for(int i = 0; i < sizeA; i++)
			{
				for(int j = 0; j < sizeA; j++)
				{
					ia[ind1] = count_chng2;
					ja[ind1] = i * sizeA + j + 1;
					
					if (count_chng2 == 1) 
						ar[ind1] = gsl_matrix_get(A, i, j);
					else
					{
						ar[ind1] = 0;
						for(int m = 0; m < sizeA; m++)
						{
							ar[ind1] = ar[ind1] + gsl_matrix_get(invA, m, i) * gsl_vector_get(x, j) * gsl_vector_get(x, k - 2);
						}
						ar[ind1] = ar[ind1] - gsl_matrix_get(invA, k - 2, i) * gsl_vector_get(x, j);
					}
					
					ind1 = ind1 + 1;        
				}
			}
			count_chng2++;
	    }
	}
	gsl_matrix_free(invA);
	

    ind1 = 1;
    glp_add_cols(lp, sizeA * sizeA);
    for(int i = 0; i < sizeA * sizeA; i++)
    {
        glp_set_col_bnds(lp, ind1, GLP_DB, -constr, constr);
        ind1 = ind1 + 1;  
    }

    /*Решаем ЗЛП*/
    ind1 = 1;
    for(int i = 0; i < sizeA; i++)
    {
        for(int j = 0; j < sizeA; j++)
        {
            glp_set_obj_coef(lp, ind1, gsl_matrix_get(B, i, j));
            ind1 = ind1 + 1;        
        }
    }

    glp_load_matrix(lp, (count_chng + 1) * sizeA * sizeA, ia, ja, ar);
    glp_simplex(lp, NULL);

    ind1 = 1;
    for(int i = 0; i < sizeA; i++)
    {
        for(int j = 0; j < sizeA; j++)
        {
            gsl_matrix_set(B, i, j, glp_get_col_prim(lp, ind1));
            ind1 = ind1 + 1;        
        }
    }

    glp_delete_prob(lp);

    return B;
}


 /* Функция для перерасчета элементов матрицы ландшафта приспособленности
  * в случае, когда в систему будут внесены изменения
  * 
  *
  *    Вход:
  *        A        - матрица ландшафта приспособленности
  *        sizeA    - порядок матрицы ландшафта приспособленности 
  *        x        - координаты положения равновесия
  *        num_view - вектор, с номерами добавленных элементов
  *        size_vec - количество элементов добавленных на текущий момент  
  *        old_f    - значение фитнеса на предыдущей итерации
  *        beta     - значение beta
  *
  *    Выход:
  *        newA - новый вид матрица ландшафта приспособленности
  */

gsl_matrix *get_new_matrix_A(gsl_matrix *A, int sizeA, gsl_vector *x, 
                             gsl_vector *num_view, int size_vec, double old_f, double beta)
{
	gsl_matrix *newA = gsl_matrix_alloc(sizeA, sizeA);
	
	gsl_vector_set(num_view, size_vec, sizeA);
	
	double new_f = 0;
	
	/*Решаем ЗЛП*/
	gsl_matrix *B = gsl_matrix_alloc(sizeA - 1, sizeA - 1); 
	gsl_matrix *A_copy = gsl_matrix_alloc(sizeA - 1, sizeA - 1);
	B = solve_lin_prog(A, x, sizeA - 1);
	gsl_matrix_memcpy(A_copy, A);
	
	/*Переписываем матрицу ландшафта */
	gsl_matrix_add(A_copy, B);
	gsl_matrix_free(B);
	
	/*Находим неподвижную точку*/
	gsl_vector *y = gsl_vector_alloc(sizeA - 1);
	y = get_freq(sizeA - 1, A_copy);
	
	/*Считаем фитнес*/
	for(int i = 0; i < sizeA - 1; i++)
		for(int j = 0; j < sizeA - 1; j++)
			new_f = new_f + gsl_matrix_get(A_copy, i, j) * gsl_vector_get(y, i) * gsl_vector_get(y, j);
	gsl_matrix_free(A_copy);        
	gsl_vector_free(y);
	
	for(int i = 0; i < sizeA - 1; i++)
		for(int j = 0; j < sizeA - 1; j++)
			gsl_matrix_set(newA, i, j, gsl_matrix_get(A, i, j));
		  
	double alpha = 1 / (gsl_vector_max(x) + 1) * beta;
		 
	for(int i = 0; i < sizeA - 1; i++)
	{
		gsl_matrix_set(newA, i, sizeA - 1, (new_f - alpha * old_f) / (1 - alpha));
		gsl_matrix_set(newA, sizeA - 1, i, new_f / (alpha * sizeA * gsl_vector_get(x, i)));
	}
	gsl_matrix_set(newA, sizeA - 1, sizeA - 1, new_f / ((1 - alpha) * sizeA));

	return newA;
}



 /* Запись данных в файлы для просмотра "глазами":
  * 
  *     - матрица ландшафта приспособленности на каждой итерации
  *     - вектор сферической нормы
  *     - средний фитнес на каждой итерации 
 */
void write_in_file(gsl_vector *A_time, int start_sizeA, int count_iter, gsl_vector * poisson_vector, 
                   gsl_vector *fitness_vec, gsl_vector *matrix_norm_vec)
{

	struct stat st = {0};
	if (stat("../Output Data", &st) == -1) 
	{    
		mkdir("../Output Data", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}
	
	if (stat("../Output Data/Data for Check", &st) == -1) 
	{    
		int flg = mkdir("../Output Data/Data for Check", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}
		
	/* матрица ландшафта приспособленности на каждой итерации*/
	int ind = 0;
	ofstream evolution_A("../Output Data/Data for Check/evolution_matrix_A.txt");
	for(int i = 0; i <= count_iter; i++)
	{
		if(gsl_vector_get(poisson_vector, i) == 1) start_sizeA++;
		for(int j = 0; j < start_sizeA; j++)
		{
		    for(int k = 0; k < start_sizeA; k++)
		    {
				evolution_A << gsl_vector_get(A_time, ind) << " ";
				ind++;
			}	
			evolution_A << endl;
		}
		evolution_A << endl;
	}
	evolution_A.close();
	
	
    /* средний фитнес на каждой итерации  */
	ofstream fitness("../Output Data/Data for Check/fitness.txt");
	for(int i = 0; i <= count_iter; i++)
		fitness << gsl_vector_get(fitness_vec, i) << endl;
	fitness.close();
	
	/* вектор сферической нормы */
	ofstream norm_A("../Output Data/Data for Check/norma_matrix_A.txt");
	for(int i = 0; i <= count_iter; i++)
		norm_A << gsl_vector_get(matrix_norm_vec, i) << endl;
	norm_A.close();
}



 /* Запись данных в файлы для отрисовки в Matlab:
  * 
  *     - вектор неподвижной точки
  *     - средний фитнес на каждой итерации
  *     - средний интегральный фитнес 
  *     - решение ОДУ
  *     - вектор сетки для быстрого времени
  *     - данные по изменениям в системе: количество элементов на каждой
  *       итерации, номера итераций на которых были внесены изменения, 
  *       значения коэффициентов beta 
  */
void write_in_file_for_Matlab(gsl_vector * poisson_vector, int count_iter, gsl_vector *num_view, int size_vec,
                              gsl_vector *fitness_vec, gsl_vector *fitness_vec_avg, int count_solve_step, gsl_vector *solve_odu_vector,
                              gsl_vector *U, int sizeU2, int sizeA, gsl_vector *count_view, gsl_vector *U_continuos, int sizeU, gsl_vector *time_vec, int count_step,
                              gsl_vector *beta_vector) 
{
	
	struct stat st = {0};
	if (stat("../Output Data", &st) == -1) 
	{    
		mkdir("../Output Data", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}
	
	if (stat("../Output Data/Data for Matlab", &st) == -1) 
	{    
		int flg = mkdir("../Output Data/Data for Matlab", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}
	
	double num;
	int num2;
	
	/*настройки для Matlab */
	ofstream set("../Output Data/Data for Matlab/settings_matlab.txt");
	set.write((char*)&count_iter, sizeof count_iter);
	set.write((char*)&size_vec, sizeof size_vec);
	set.write((char*)&count_solve_step, sizeof count_solve_step);
	set.write((char*)&sizeA, sizeof sizeA);
	set.write((char*)&count_step, sizeof count_step);
	set.close();

	
	/*информацию по внесению изменений в ситему*/
	ofstream poisson("../Output Data/Data for Matlab/poisson_matlab.txt", ios::binary | ios::out);
	
	for(int i = 0; i <= count_iter; i++)
    {
	    num2 = gsl_vector_get(poisson_vector, i); 
		poisson.write((char*)&num2, sizeof num2);
	}
	poisson.close();
	
	/*значения коэффициента beta*/
	ofstream beta("../Output Data/Data for Matlab/beta_matlab.txt", ios::binary | ios::out);
	
	for(int i = 0; i <= count_iter; i++)
    {
	    num = gsl_vector_get(beta_vector, i); 
		beta.write((char*)&num, sizeof num);
	}
	beta.close();
	
	/*Количество добавлений*/
	ofstream view("../Output Data/Data for Matlab/num_view_matlab.txt", ios::binary | ios::out);
	
	for(int i = 0; i < size_vec; i++)
    {
	    num2 = gsl_vector_get(num_view, i); 
		view.write((char*)&num2, sizeof num2);
	}
	view.close();
	
	/* средний фитнес на каждой итерации */
	ofstream fitn("../Output Data/Data for Matlab/fitness_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i <= count_iter; i++)
	{
		num = gsl_vector_get(fitness_vec, i);
		fitn.write((char*)&num, sizeof num);
	}				
	fitn.close();
	
	/* средний интегральный фитнес */
	ofstream fitn_avg("../Output Data/Data for Matlab/fitness_avg_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i < count_solve_step; i++)
	{
		num = gsl_vector_get(fitness_vec_avg, i);
		fitn_avg.write((char*)&num, sizeof num);
	}				
	fitn_avg.close();
	
	/*номера итераций, на которых решали ОДУ*/
	ofstream solve_odu("../Output Data/Data for Matlab/solve_odu_matlab.txt", ios::binary | ios::out);
	
	for(int i = 0; i <= count_iter; i++)
    {
	    num2 = gsl_vector_get(solve_odu_vector, i); 
		solve_odu.write((char*)&num2, sizeof num2);
	}
	solve_odu.close();
	
	/* вектор неподвижной точки */
	ofstream freq("../Output Data/Data for Matlab/freqType_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i < sizeU2; i++)
	{
		num = gsl_vector_get(U, i); 
		freq.write((char*)&num, sizeof num);
	}
	freq.close();
	
	/*вектор кол-ва элементов на каждой итерации*/
	ofstream view2("../Output Data/Data for Matlab/count_view_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i <= count_iter; i++)
	{
		num2 = gsl_vector_get(count_view, i); 
		view2.write((char*)&num2, sizeof num2);
	}
	view2.close();
	
    /* решение ОДУ */
	ofstream freq_cont("../Output Data/Data for Matlab/freqType_continuos_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i < sizeU; i++)
	{
	    num = gsl_vector_get(U_continuos, i);
		freq_cont.write((char*)&num, sizeof num);
	}
	freq_cont.close();
	
	/* вектор сетки для быстрого времени */
	ofstream time("../Output Data/Data for Matlab/time_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i <= count_step; i++)
	{
		num = gsl_vector_get(time_vec, i);
		time.write((char*)&num, sizeof num);
	} 
	time.close();
}


int main(int *argc, char **argv)
{   
    /* Ввод данных с клавиатуры:
     *     - порядок матрицы ландшафта приспособленности 
     *     - количество итераций эволюции (сколько раз будем решать ЗЛП)
     *     - горизонт для решения ОДУ (некоторое число T. Тогда ОДУ будет решаться на [0, T])
     *     - сетка для решения ОДУ (шаг для разбиения отрезка [0, T])
     *     - как часто будем решать ОДУ (натуральное число - шаг по итерациям) 
     *     - параметры Пуассоновского процесса
     *     - максимальное число изменение, которое может быть внесено в исходную систему
     */
    int sizeA, count_iter, solve_step, count_new_type;
    double t1, h, lambda;
    /*порядок матрицы ландшафта приспособленности */
    cout << "Введите порядок матрицы A "; cin >> sizeA; cout << endl; int start_sizeA = sizeA;
    /*количество итераций эволюции (сколько раз будем решать ЗЛП)*/
    cout << "Введите количество итераций эволюции (сколько раз будем решать ЗЛП) "; cin >> count_iter; cout << endl;
    /*горизонт для решения ОДУ (некоторое число T. Тогда ОДУ будет решаться на [0, T])*/
    cout << "Введите горизонт для решения ОДУ (некоторое число T. Тогда ОДУ будет решаться на [0, T]) "; cin >> t1; cout << endl;
    /*сетка для решения ОДУ (количество точек для разбиения отрезка [0, T])*/
    cout << "Введите шаг для разбиения отрезка [0, T] "; cin >> h; cout << endl;
    /*как часто будем решать ОДУ (натуральное число - шаг по итерациям)*/
    cout << "Введите шаг по итерациям, с которым будем решать ОДУ (натуральное число) "; cin >> solve_step; cout << endl;  
    /*вероятность внесения изменений в систему*/
    cout << "Введите вероятность внесения изменений в систему на каждой итерации "; cin >> lambda; cout << endl; 
    /*максимальное число изменений в системе*/
    cout << "Введите максимальное число изменений в системе "; cin >> count_new_type; cout << endl;     
    

    /*Считываем матрицу ландшафта приспособленности и начальную точку для решения ОДУ*/
    gsl_matrix *A = gsl_matrix_alloc(sizeA, sizeA);
    gsl_vector *u0 = gsl_vector_alloc(sizeA);
    
    double buff;
    ifstream fin_A("../Input Data/Matrix_A.txt");
    ifstream fin_u0("../Input Data/u0.txt");
    for(int i = 0; i < sizeA; i++)
    {
		fin_u0 >> buff; gsl_vector_set(u0, i, buff);
        for(int j = 0; j < sizeA; j++)
        {
            fin_A >> buff;   
            gsl_matrix_set(A, i, j, buff);         
        }
    }     
    fin_A.close(); fin_u0.close();
    
    /*Сразу определим итерации, на которых в систему будут внесены изменения*/
    gsl_vector *poisson_vector;
    poisson_vector = poisson_new_element(count_iter, lambda, count_new_type);
    
    /* После этого определим коэффициенты beta для итераций, 
     * когда будут добавлены новые виды
     */
    gsl_vector *beta_vector;
    beta_vector = get_beta_vector(count_iter, poisson_vector);
    
    /*определим на каких итерациях будем решать ОДУ*/
    gsl_vector *solve_odu_vector;
    solve_odu_vector = get_step_odu(count_iter, solve_step, poisson_vector);
    
    /*Данные для решения ОДУ*/
    int count_solve_step = 0, sizeA_max = sizeA, sizeU = 0, sizeU2 = 0, sizeA2 = 0;
    double count_step, t0;
    modf(t1 / h, &count_step);
    
    for(int i = 0; i <= count_iter; i++)
    {
		if(gsl_vector_get(solve_odu_vector, i) == 1)
		{ 
			count_solve_step++;
			if(gsl_vector_get(poisson_vector, i) == 1) sizeA_max++;
			sizeU = sizeU + sizeA_max * (count_step + 1); 
		} 
		sizeU2 = sizeU2 + sizeA_max;
		sizeA2 = sizeA2 + sizeA_max * sizeA_max;
	}
	gsl_vector *U_continuos = gsl_vector_alloc(sizeU);
	
	gsl_odeiv2_driver * d;
    gsl_odeiv2_system sys;
    
    gsl_vector *time_vec = gsl_vector_alloc(count_step + 1); 
    for(int i = 0; i <= count_step; i++)
        gsl_vector_set(time_vec, i, h * i);
	
	
    /*Выходные данные*/
    //матрица значений положений равновесия на каждой итерации эволюции
    gsl_vector *U = gsl_vector_alloc(sizeU2);
    //вектор значений среднего фитнеса на каждой итерации эволюции
    gsl_vector *fitness_vec = gsl_vector_alloc(count_iter + 1);
    //вектор значений среднего интегрального фитнеса
    gsl_vector *fitness_vec_avg = gsl_vector_alloc(count_solve_step);
    //матрица для хранения значений матрицы ландшафта приспособленности на каждой итерации эволюции
    gsl_vector *A_time = gsl_vector_alloc(sizeA2); 
    //ветор значений сферической нормы матрицы ландшафта приспособленности на каждой итерации эволюции
    gsl_vector *matrix_norm_vec = gsl_vector_alloc(count_iter + 1);
  
    //Вектор для хранения номеров новых элементов 
    gsl_vector *num_view;
    int size_vec = 0;
    for(int i = 0; i <= count_iter; i++)
        if(gsl_vector_get(poisson_vector, i) > 0) size_vec++;   
        
    if(size_vec == 0) size_vec++;    
    num_view = gsl_vector_alloc(size_vec);
    size_vec = 0;
    
    /* Вектор, в котором будет храниться кол-во элементов на каждой итерации*/
    gsl_vector *count_view = gsl_vector_alloc(count_iter + 1);

    gsl_vector *x;    
    gsl_matrix *B;       
    int sizeU2_ = 0, sizeA2_ = 0, sizeU_ = 0, count_solve_step2 = 0;
    
    /*Основной процесс эволюции ландшафта приспособленности*/
    for(int i = 0; i <= count_iter; i++)
    {
		cout << "I = " << i << endl << endl;
		    
        /*Находим неподвижную точку*/
        x = get_freq(sizeA, A);
        
        if(gsl_vector_min(x) >= 0)
        {
			/*Записываем количество видов, которое будет на данной итерации*/
			gsl_vector_set(count_view, i, sizeA);
			
			/*Сохраняем найденную неподвижную точку*/
			for(int k = 0; k < sizeA; k++)
			{
				gsl_vector_set(U, sizeU2_, gsl_vector_get(x, k));
				sizeU2_++;
			}  

            /*Вычисляем средний фитнес*/
			gsl_vector_set(fitness_vec, i, 0);
			for(int k = 0; k < sizeA; k++)
				for(int j = 0; j < sizeA; j++)
					gsl_vector_set(fitness_vec, i, gsl_vector_get(fitness_vec, i) + gsl_matrix_get(A, k, j) * gsl_vector_get(x, k) * gsl_vector_get(x, j));
					
			/*Вычисляем сферическую норму*/
			gsl_vector_set(matrix_norm_vec, i, 0);
			for(int k = 0; k < sizeA; k++)
				for(int j = 0; j < sizeA; j++)
				    gsl_vector_set(matrix_norm_vec, i, gsl_vector_get(matrix_norm_vec, i) + gsl_matrix_get(A, k, j) * gsl_matrix_get(A, k, j));
	 
			/*Сохраняем вид матрицы ландшафта приспособленности на каждой итерации*/
			for(int k = 0; k < sizeA; k++)
				for(int j = 0; j < sizeA; j++)
				{
					gsl_vector_set(A_time, sizeA2_, gsl_matrix_get(A, k, j));
					sizeA2_++;
				}
					
			/*Смотрим нужно ли решать ОДУ*/
			if (gsl_vector_get(solve_odu_vector, i) == 1)
			{
				double y[sizeA];
				sys = {func, jac, sizeA, A};
				d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0);
			
				for(int k = 0; k < sizeA; k++) 
				{
					y[k] = gsl_vector_get(u0, k);
					gsl_vector_set(U_continuos, sizeU_, y[k]);
					sizeU_++;
				}
			
				t0 = 0.0;
				for(int k = 0; k < count_step; k++)
				{
					double ti = h * (k + 1);
					int status = gsl_odeiv2_driver_apply (d, &t0, ti, y);
					if (status != GSL_SUCCESS)
					{
						printf ("error, return value=%d\n", status);
						break;
					}
					for(int j = 0; j < sizeA; j++){
					    gsl_vector_set(U_continuos, sizeU_, y[j]); 
					    sizeU_++;
					}
				}
				
				/*Вычисляем средний интегральный фитнес*/
				gsl_vector_set(fitness_vec_avg, count_solve_step2, get_avg_integral_fitness(U_continuos, A, sizeA, count_step, sizeU_));
				
				gsl_odeiv2_driver_free(d);  
				count_solve_step2++;
			}      
	    }
	    
	    else
	    {
		    cout << "COMPONENTS LESS 0" << endl;
		    
		    for(int l = 0; l < sizeA; l++) cout << gsl_vector_get(x, l) << " ";
		    cout << endl << endl;
		    break;    	
		}
		 
		if(i < count_iter)
		{      
			/* Проверяем будет ли система изменена */
			if(gsl_vector_get(poisson_vector, (i + 1)) > 0)
			{			
					
				sizeA++;
				/*перезаписываем матрицу ландшафта*/
				gsl_matrix *newA = gsl_matrix_alloc(sizeA, sizeA);
				newA = get_new_matrix_A(A, sizeA, x, num_view, size_vec, gsl_vector_get(fitness_vec, i), gsl_vector_get(beta_vector, i + 1));					
				gsl_matrix_free(A);
				A = gsl_matrix_alloc(sizeA, sizeA);
				gsl_matrix_memcpy(A, newA);
				gsl_matrix_free(newA);
				
				/*перезаписываем вектор неподвижной точки*/
				double alpha = 1 / (gsl_vector_max(x) + 1) * gsl_vector_get(beta_vector, i + 1);
				
				gsl_vector *newX = gsl_vector_alloc(sizeA);
				for(int k = 0; k < sizeA - 1; k++)
					gsl_vector_set(newX, k, gsl_vector_get(x, k) * alpha);
				gsl_vector_set(newX, sizeA - 1, (1 - alpha));
				
				gsl_vector_free(x);
				x = gsl_vector_alloc(sizeA);
				gsl_vector_memcpy(x, newX);
				gsl_vector_free(newX);
				
				/*перезаписываем вектор начальных данных*/
				gsl_vector *new_u0 = gsl_vector_alloc(sizeA);
				gsl_vector_set(new_u0, (sizeA - 1), 0);
				for(int k = 0; k < (sizeA - 1); k++)
					gsl_vector_set(new_u0, k, 0.9 * gsl_vector_get(u0, k));
				gsl_vector_set(new_u0, (sizeA - 1), 0.1);
				
				gsl_vector_free(u0);
				u0 = gsl_vector_alloc(sizeA);
				gsl_vector_memcpy(u0, new_u0);
				gsl_vector_free(new_u0);
				size_vec++;			
			}
	    } 		
		
		if(i < count_iter)
		{ 
			if(gsl_vector_get(poisson_vector, (i + 1)) == 0)
			{
				/* Решаем ЗЛП */
				B = solve_lin_prog(A, x, sizeA);
				gsl_matrix_add(A, B);
				
				gsl_matrix_free(B);
			}
	    }
        gsl_vector_free(x);
    }   
   
    write_in_file(A_time, start_sizeA, count_iter, poisson_vector, fitness_vec, matrix_norm_vec);
    write_in_file_for_Matlab(poisson_vector, count_iter, num_view, size_vec, fitness_vec, fitness_vec_avg, 
                             count_solve_step, solve_odu_vector, U, sizeU2, sizeA, count_view, U_continuos, sizeU, time_vec, count_step, beta_vector);
    
    gsl_matrix_free(A);
    gsl_vector_free(u0);
    gsl_vector_free(poisson_vector);
    gsl_vector_free(beta_vector);
    gsl_vector_free(solve_odu_vector);
    gsl_vector_free(U_continuos);
    gsl_vector_free(time_vec); 
    gsl_vector_free(U);
    gsl_vector_free(fitness_vec);
    gsl_vector_free(fitness_vec_avg);
    gsl_vector_free(A_time);
    gsl_vector_free(matrix_norm_vec);
    gsl_vector_free(num_view);
    gsl_vector_free(count_view);
            
    return 0;
}



