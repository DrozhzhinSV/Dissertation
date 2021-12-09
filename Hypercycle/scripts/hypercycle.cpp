#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cmath> 
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

const double EPS = 2.0E-2;
const double constr = 1.0E-3;

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


 /* Функция для расчета коэффициента эгоистичности: доля ресурсов каждого вида,
  * затраченная на себя 
  *
  *     Вход:
  *         sizeA - размер матрицы ландшафта приспособленности
  *         A     - матрица ландшафта приспособленности
  *
  *     Выход:
  *         selfish_factor - вектор коэффициентов эгоистичности
 */
 
gsl_vector *get_selfish_factor(int sizeA, gsl_matrix *A)
{
	gsl_vector *selfish_factor = gsl_vector_alloc(sizeA);
	
	double denominator, numenator;
	for(int i = 0; i < sizeA; i++)
	{
		denominator = 0;
		for(int j = 0; j < sizeA; j++)
		{
			denominator = denominator + gsl_matrix_get(A, j, i) * gsl_matrix_get(A, j, i);
			if(j == i) numenator = gsl_matrix_get(A, i, i) * gsl_matrix_get(A, i, i);
		}
		gsl_vector_set(selfish_factor, i, numenator / denominator);
	}
	
	return selfish_factor;
}


 /* Функция для расчета коэффициента, определяющего какая доля ресурсов, 
  * приходящаяся на каждый вид, идет от него самого 
  *
  *     Вход:
  *         sizeA - размер матрицы ландшафта приспособленности
  *         A     - матрица ландшафта приспособленности
  *
  *     Выход:
  *         selfish_factor - вектор коэффициентов эгоистичности
 */
 
gsl_vector *get_self_influence_factor(int sizeA, gsl_matrix *A)
{
	gsl_vector *self_influence_factor = gsl_vector_alloc(sizeA);
	
	double denominator, numenator;
	for(int i = 0; i < sizeA; i++)
	{
		denominator = 0;
		for(int j = 0; j < sizeA; j++)
		{
			denominator = denominator + gsl_matrix_get(A, i, j) * gsl_matrix_get(A, i, j);
			if(j == i) numenator = gsl_matrix_get(A, i, i) * gsl_matrix_get(A, i, i);
		}
		gsl_vector_set(self_influence_factor, i, numenator / denominator);
	}
	
	return self_influence_factor;
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
    for(int i = 0; i < sizeA; i++)
    {
        for(int j = 0; j < sizeA; j++)
        {
            glp_set_col_bnds(lp, ind1, GLP_DB, -constr, constr);
            ind1 = ind1 + 1;        
        }
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

/* Функция для нахождения собственных значений и собственного вектора:
 * 
 * Печатает результаты на экран
 */
void get_eigen_value(gsl_matrix *A, gsl_vector *x, int sizeA)
{
	double data[sizeA * sizeA];
	
	for(int i = 0; i < sizeA; i++)
    {
		for(int j = 0; j < sizeA; j++)
		{
			data[i * sizeA +  j] = gsl_vector_get(x, i) * gsl_matrix_get(A, i, j);
			for(int k = 0; k < sizeA; k++)
			    data[i * sizeA + j] = data[i * sizeA + j] - gsl_vector_get(x, i) * gsl_vector_get(x, k) * (gsl_matrix_get(A, k, j) + gsl_matrix_get(A, j, k));
			
			if (i == j)
			{
				for(int k = 0; k < sizeA; k++)
				{
					data[i * sizeA + i] = data[i * sizeA + i] + gsl_vector_get(x, k) * gsl_matrix_get(A, i, k);
				    for(int l = 0; l < sizeA; l++)
				       data[i * sizeA + i] = data[i * sizeA + i] - gsl_vector_get(x, k) * gsl_vector_get(x, l) * gsl_matrix_get(A, k, l);
				} 
			} 
		}
	}
	
	gsl_matrix_view m = gsl_matrix_view_array (data, sizeA, sizeA);
    gsl_vector_complex *eval = gsl_vector_complex_alloc (sizeA);
    gsl_matrix_complex *evec = gsl_matrix_complex_alloc (sizeA, sizeA);
    gsl_eigen_nonsymmv_workspace * w = gsl_eigen_nonsymmv_alloc (sizeA);
    gsl_eigen_nonsymmv (&m.matrix, eval, evec, w);
    gsl_eigen_nonsymmv_free (w);
    gsl_eigen_nonsymmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_DESC);
    
    {
		int i, j;
		for (i = 0; i < sizeA; i++)
		{
			gsl_complex eval_i = gsl_vector_complex_get (eval, i);
			gsl_vector_complex_view evec_i = gsl_matrix_complex_column (evec, i);
			printf ("eigenvalue = %g + %gi\n", GSL_REAL(eval_i), GSL_IMAG(eval_i));
			printf ("eigenvector = \n");
			for (j = 0; j < sizeA; ++j)
			{
				gsl_complex z = gsl_vector_complex_get(&evec_i.vector, j);
				printf("%g + %gi\n", GSL_REAL(z), GSL_IMAG(z));
			}
		}
	}
	gsl_vector_complex_free(eval);
	gsl_matrix_complex_free(evec);
	
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
double get_avg_integral_fitness(gsl_matrix *U_continuos, gsl_matrix *A, int sizeA, int count_solve_step2, double count_step)
{
	double s, f = 0;
	for(int i = 0; i <= count_step; i++)
	{
		s = 0;
		for(int j = 0; j < sizeA; j++)
		    for(int k = 0; k < sizeA; k++)
		        s = s + gsl_matrix_get(A, j, k) * gsl_matrix_get(U_continuos, count_solve_step2, j * (count_step + 1) + i) * gsl_matrix_get(U_continuos, count_solve_step2, k * (count_step + 1) + i);
		
		if ((i ==0) || (i == count_step)) s = s / 2;
		f = f + s;
    }
	   
	f = f / count_step;
	
	return f;
}


 /* Запись данных в файлы для просмотра "глазами":
  * 
  *     - матрица ландшафта приспособленности на каждой итерации
  *     - вектор сферической нормы
  *     - средний фитнес на каждой итерации 
 */
void write_in_file(int sizeA, int count_iter, gsl_matrix *A_time, gsl_vector *matrix_norm_vec, gsl_vector *fitness_vec)
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
	ofstream evolution_A("../Output Data/Data for Check/evolution_matrix_A.txt");
	for(int i = 0; i <= count_iter; i++)
	{
		for(int j = 0; j < sizeA; j++)
		{
			for(int k = 0; k < sizeA; k++)
			{
				evolution_A << gsl_matrix_get(A_time, j * sizeA + k, i) << " ";
			}
			evolution_A << endl;
		}
		evolution_A << endl;
	}
	evolution_A.close();
	
	/* вектор сферической нормы */
	ofstream norm_A("../Output Data/Data for Check/norma_matrix_A.txt");
	for(int i = 0; i <= count_iter; i++)
		norm_A << gsl_vector_get(matrix_norm_vec, i) << endl;
	norm_A.close();
	
	/* средний фитнес на каждой итерации  */
	ofstream fitness("../Output Data/Data for Check/fitness.txt");
	for(int i = 0; i <= count_iter; i++)
		fitness << gsl_vector_get(fitness_vec, i) << endl;
	fitness.close();
}


 /* Запись данных в файлы для отрисовки в Matlab:
  * 
  *     - вектор неподвижной точки
  *     - средний фитнес на каждой итерации
  *     - средний интегральный фитнес 
  *     - решение ОДУ
  *     - вектор сетки для быстрого времени
  *     - коэффициенты эгоистичности 
 */
void write_in_file_for_Matlab(int sizeA, int count_iter, double count_step, int solve_step, double count_solve_step, 
                              gsl_matrix *U, gsl_vector *fitness_vec, gsl_vector *fitness_vec_avg, gsl_matrix *U_continuos, 
                              gsl_vector *time_vec, gsl_matrix *coef_self_factor, gsl_matrix *coef_self_influence_factor)
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
	
	/* вектор неподвижной точки */
	ofstream freq("../Output Data/Data for Matlab/freqType_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i < sizeA; i++)
	{
		for(int j = 0; j <= count_iter; j++)
		{
			num = gsl_matrix_get(U, i, j); 
			freq.write((char*)&num, sizeof num);
		}
	}
	freq.close();
	
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
	
    /* решение ОДУ */
	ofstream freq_cont("../Output Data/Data for Matlab/freqType_continuos_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i < count_solve_step; i++)
	{
		for(int j = 0; j < sizeA * (count_step + 1); j++)
		{
				num = gsl_matrix_get(U_continuos, i, j);
				freq_cont.write((char*)&num, sizeof num);
		}
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
	
	/* коэффициенты эгоистичности */
	ofstream self("../Output Data/Data for Matlab/coef_self_factor.txt", ios::binary | ios::out);
	for(int i = 0; i < sizeA; i++)
	{
		for(int j = 0; j <= count_iter; j++)
		{
			num = gsl_matrix_get(coef_self_factor, i, j); 
			self.write((char*)&num, sizeof num);
		}
	}
	self.close();
	
	ofstream self_infl("../Output Data/Data for Matlab/coef_self_influence_factor.txt", ios::binary | ios::out);
	for(int i = 0; i < sizeA; i++)
	{
		for(int j = 0; j <= count_iter; j++)
		{
			num = gsl_matrix_get(coef_self_influence_factor, i, j); 
			self_infl.write((char*)&num, sizeof num);
		}
	}
	self_infl.close();
	
	/* настройки для Matlab */
	ofstream set("../Output Data/Data for Matlab/settings_matlab.txt");
	set.write((char*)&sizeA, sizeof sizeA);
	set.write((char*)&count_iter, sizeof count_iter);
	set.write((char*)&count_step, sizeof count_step);
	set.write((char*)&solve_step, sizeof solve_step);
	set.write((char*)&count_solve_step, sizeof count_solve_step);
	set.close();
}


int main(int *argc, char **argv)
{
   /* Ввод данных с клавиатуры:
    *     - порядок матрицы ландшафта приспособленности 
    *     - количество итераций эволюции (сколько раз будем решать ЗЛП)
    *     - горизонт для решения ОДУ (некоторое число T. Тогда ОДУ будет решаться на [0, T])
    *     - сетка для решения ОДУ (шаг для разбиения отрезка [0, T])
    *     - как часто будем решать ОДУ (натуральное число - шаг по итерациям) 
   */
    int sizeA, count_iter, solve_step;
    double t1, h;
    /*порядок матрицы ландшафта приспособленности */
    cout << "Введите порядок матрицы A "; cin >> sizeA; cout << endl;
    /*количество итераций эволюции (сколько раз будем решать ЗЛП)*/
    cout << "Введите количество итераций эволюции (сколько раз будем решать ЗЛП) "; cin >> count_iter; cout << endl;
    /*горизонт для решения ОДУ (некоторое число T. Тогда ОДУ будет решаться на [0, T])*/
    cout << "Введите горизонт для решения ОДУ (некоторое число T. Тогда ОДУ будет решаться на [0, T]) "; cin >> t1; cout << endl;
    /*сетка для решения ОДУ (количество точек для разбиения отрезка [0, T])*/
    cout << "Введите шаг для разбиения отрезка [0, T] "; cin >> h; cout << endl;
    /*как часто будем решать ОДУ (натуральное число - шаг по итерациям)*/
    cout << "Введите шаг по итерациям, с которым будем решать ОДУ (натуральное число) "; cin >> solve_step; cout << endl;    

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
            fin_A         >> buff;   gsl_matrix_set(A,               i, j,    buff);       
        }
    }        
    fin_A.close(); fin_u0.close();
    

    /*Выходные данные*/
    //матрица значений положений равновесия на каждой итерации эволюции
    gsl_matrix *U = gsl_matrix_alloc(sizeA, count_iter + 1);
    //вектор значений среднего фитнеса на каждой итерации эволюции
    gsl_vector *fitness_vec = gsl_vector_alloc(count_iter + 1);
    //матрица для хранения значений матрицы ландшафта приспособленности на каждой итерации эволюции
    gsl_matrix *A_time = gsl_matrix_alloc(sizeA * sizeA, count_iter + 1); 
    //ветор значений сферической нормы матрицы ландшафта приспособленности на каждой итерации эволюции
    gsl_vector *matrix_norm_vec = gsl_vector_alloc(count_iter + 1);
    
    //векторы для коэффициентов эгоистичности
    gsl_matrix *coef_self_factor = gsl_matrix_alloc(sizeA, count_iter + 1);
    gsl_matrix *coef_self_influence_factor = gsl_matrix_alloc(sizeA, count_iter + 1);
    
    /*Данные для решения ОДУ*/
    double y[sizeA], count_step, t0, count_solve_step;
    modf(t1 / h, &count_step);
    modf((count_iter + 1) / solve_step, &count_solve_step); 
    if((count_solve_step * solve_step == (count_iter + 1)) || (count_iter == 0)) count_solve_step++;
    else count_solve_step = count_solve_step + 2;
    
    int solve_step2 = 1, count_solve_step2 = 0;
    //матрица для хранения решений ОДУ
    gsl_matrix *U_continuos = gsl_matrix_alloc(count_solve_step, sizeA * (count_step + 1));
    gsl_odeiv2_driver * d;
    gsl_odeiv2_system sys;
    
    //вектор значений среднего интегрального фитнеса
    gsl_vector *fitness_vec_avg = gsl_vector_alloc(count_solve_step);
    
    gsl_vector *time_vec = gsl_vector_alloc(count_step + 1); 
    for(int i = 0; i <= count_step; i++)
        gsl_vector_set(time_vec, i, h * i);
    
    gsl_vector *x;    
    gsl_matrix *B; 
    

    
    /* Основной процесс эволюции ландшафта приспособленности*/
    for(int i = 0; i <= count_iter; i++)
    {
		cout << "I = " << i << endl << endl; /* Печать номера итерации */
     
        /*Находим неподвижную точку*/
        x = get_freq(sizeA, A);  
        
        /*Находим коэффициенты эгоистичности*/
        gsl_matrix_set_col(coef_self_factor, i, get_selfish_factor(sizeA, A));
        gsl_matrix_set_col(coef_self_influence_factor, i, get_self_influence_factor(sizeA, A));
        
        /*Рассчитываем собственные значения*/
        //get_eigen_value(A, x, sizeA);
        
        /*Смотрим, что все компоненты неподвижной точки принадлежат единичному симплексу*/
        if(gsl_vector_min(x) >= 0)
        {  
			/*Сохраняем найденную неподвижную точку*/
			gsl_matrix_set_col(U, i, x);  

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
					gsl_matrix_set(A_time, k * sizeA + j, i, gsl_matrix_get(A, k, j));
						
			/*Смотрим нужно ли решать ОДУ*/
			if (i == 0 || i == count_iter || solve_step2 == solve_step)
			{
				sys = {func, jac, sizeA, A};
				d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0);
			
				for(int k = 0; k < sizeA; k++) 
				{
					y[k] = gsl_vector_get(u0, k);
					gsl_matrix_set(U_continuos, count_solve_step2, k * (count_step + 1), y[k]);
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
					for(int j = 0; j < sizeA; j++)
					    gsl_matrix_set(U_continuos, count_solve_step2, j * (count_step + 1) + k + 1, y[j]); 
				}
				
				/*Вычисляем средний интегральный фитнес*/
				gsl_vector_set(fitness_vec_avg, count_solve_step2, get_avg_integral_fitness(U_continuos, A, sizeA, count_solve_step2, count_step));
				
				gsl_odeiv2_driver_free(d);  
				count_solve_step2++;
			}      
			solve_step2++;
			if (solve_step2 > solve_step) solve_step2 = 1;
	    }
	    
	    else
	    {
		    count_iter = i - 1;
		    count_solve_step = count_solve_step2;
		    cout << "COMPONENTS LESS 0" << endl;
		    for(int ii = 0; ii < sizeA; ii++)
		        cout << gsl_vector_get(x, ii) << "  ";
		    break;    	
		}
		      
        /*Решаем ЗЛП*/
        B = solve_lin_prog(A, x, sizeA);
        /*Переписываем матрицу ландшафта приспособленности*/
        gsl_matrix_add(A, B);
        gsl_matrix_free(B);
        gsl_vector_free(x);
    }   
    
    
    write_in_file(sizeA, count_iter, A_time, matrix_norm_vec, fitness_vec);
	write_in_file_for_Matlab(sizeA, count_iter, count_step, solve_step, count_solve_step, U, fitness_vec, fitness_vec_avg, U_continuos, time_vec,
	                         coef_self_factor, coef_self_influence_factor); 

    gsl_matrix_free(A);
    gsl_vector_free(matrix_norm_vec);
    gsl_matrix_free(U);
    gsl_matrix_free(A_time);
    gsl_matrix_free(U_continuos);
    gsl_vector_free(fitness_vec);
    gsl_vector_free(fitness_vec_avg);
    gsl_vector_free(time_vec);
    gsl_vector_free(u0);
    gsl_matrix_free(coef_self_factor);
    gsl_matrix_free(coef_self_influence_factor);
            
    return 0;
}
