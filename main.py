
import numpy as np
from tqdm import tqdm # Progress bar
from mpmath import mp
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.animation as animation


# гравитационная постоянная
G = 1
M_Sun = 1.98840987e+30
M_Jup = 1.8981246e+27


def vector_by_N(vec, N):
    """
    функция разбивает вектор vec для N, представленный как сплошной массив,
    на вектор размера (N, 3) - для каждого тела отдельный вектор
    """
    return mp.matrix(np.reshape(vec, (N, 3)))

def vector_to_N(vec):
    """
    функция обратная к vector_by_N
    """
    # странно, но функция np.array преобразует mp.matrix
    # в сплошной массив, не зависимо от размерности vec
    N = len(vec)
    return mp.matrix(np.array(vec.tolist()).reshape(N*3))


def T(p, m):
    """
    Кинетическая энергия
    :param p: вектор импульсов
    :param m: вектор масс
    :return: значение кинетической энергии
    """
    N = len(m)
    p_by_n = vector_by_N(p, N)
    T = 0
    for i in range(N):
        T += mp.norm(p_by_n[i, :]) ** mp.mpf('2') / (mp.mpf('2')*m[i])
    return T


def U(q, m):
    """
    Потенциальная энергия
    :param q: вектор в конфигурационном пространстве
    :param m: вектор масс
    :return: Значение потенциальной энергии
    """
    N = len(m)
    q_by_n = vector_by_N(p, N)
    U = 0
    for i in range(N-1):
        for j in range(i+1, N):
            U += G*m[i]*m[j] / mp.norm(q_by_n[i,:] - q_by_n[j,:])
    return U


def dT(p, m):
    """
    Производная кинетической энергии по импульсу
    :param p: вектор импульса
    :param m: вектор масс
    :return: Вектор (dT/dp_i), i = 1,...,n, где n-размерность конфигурационного пространства
    """
    N = len(m)
    p_by_n = vector_by_N(p, N)
    dT = []
    for i in range(N):
        one_dT = p_by_n[i, :] / m[i]
        dT.append(one_dT)

    dT = mp.matrix(np.array(dT))
    dT = vector_to_N(dT)
    return dT


def dU(q, m):
    """
    Производная потенцильной энергии по импульсу
    :param q: вектор координат
    :param m: вектор масс
    :return: Вектор (dU/dq_i), i = 1,...,n, где n-размерность конфигурационного пространства
    """
    N = len(m)
    q_by_n = vector_by_N(q, N)
    dU = []
    for i in range(N):
        dUi = mp.matrix([['0', '0', '0']])
        for j in range(N):
            if i == j:
                continue
            dUi += G*m[i]*m[j] * (q_by_n[i,:] - q_by_n[j,:]) / mp.norm((q_by_n[i,:] - q_by_n[j,:]))**mp.mpf('3')
        dU.append(dUi)

    dU = mp.matrix(np.array(dU))
    dU = vector_to_N(dU)
    return dU



def stormlet_verlet_explicit(q, p, m, step, n=1000, to_file=False, filename=None):
    """
    Метод Штормер-Верле, симплектический второго порядка
    :param q: вектор начальных данных по координатам
    :param p: вектор начальных данных по импульсу
    :param step: размер шага интегрирования по времени
    :param n: количество шагов для интегрирования
    :param to_file: True, если надо записывать в файл
    :param filename: имя файла для вывода результата
    :return: значения координаты и скорости на последнем шаге
    """
    if to_file:
        assert filename is not None, 'Необходимо указать имя файла'
        f = open(filename, 'w')
        original_stdout = sys.stdout
        sys.stdout = f
    for i in tqdm(range(n)):
        p_half2 = p - step / mp.mpf('2') * dU(q, m)
        q = q + step * dT(p_half2, m)
        p_n1 = p_half2 - step / mp.mpf('2') * dU(q, m)
        p = p_n1

        if to_file:
            print(i * step, ' '.join(map(lambda x: str(x[0]), q.tolist())),
                  ' '.join(map(lambda x: str(x[0]), p.tolist())))
    if to_file:
        f.close()
        sys.stdout = original_stdout
    q_3d = mp.matrix([q[0], q[1], q[2]])
    p_3d = mp.matrix([p[0], p[1], p[2]])

    return q_3d, p_3d


def to_alenas_format(file_name):
    data_frame = pd.read_csv(file_name, delimiter=' ', header=None)
    data_frame.to_csv('t.txt', header=False, columns=[0], index=False)
    j_body = 1
    for i in range(1, int((data_frame.shape[1] - 1) / 2), 3):
        data_frame.to_csv(f'XYZ-{j_body}.txt', sep=' ', index=False, header=False, columns=[i, i+1, i+2])
        j_body += 1
    j_body = 1
    for k in range(int((data_frame.shape[1] - 1) / 2 + 1), data_frame.shape[1]-1, 3):
        data_frame.to_csv(f'VxVyVz-{j_body}.txt', sep=' ', index=False, header=False, columns=[k, k + 1, k + 2])
        j_body += 1



def get_init(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = list(map(lambda x: x.replace('\n', ''), lines))

    assert len(lines) % 7 == 0
    bodies = np.array(list(zip(*[iter(lines)]*7)))

    m, q, v = bodies[:, 0], bodies[:, 1:4], bodies[:, 4:7]
    N = len(m)
    m = mp.matrix(m)
    q = mp.matrix(q)
    v = mp.matrix(v)
    p = []
    for i in range(N):
        p.append(v[i, :]*m[i])
    p = mp.matrix(np.array(p))


    q = vector_to_N(q)
    p = vector_to_N(p)

    return q, p, m


if __name__ == '__main__':
    import argparse

    # parser = argparse.ArgumentParser(description='Provide step')
    # parser.add_argument('--step', type=str, default='0.1',
    #                 help='step of integration')
    # args = parser.parse_args()


    # init_pos, init_vel, masses = get_init('initial.dat')
    mp.dps = 30
    masses = mp.matrix([mp.mpf('0.333333333333333333'), mp.mpf('0.333333333333333333'), mp.mpf('0.333333333333333333')])
    x_2_dot = mp.mpf('0.749442191077792')
    y_2_dot = mp.mpf('1.1501789857502275')
    init_vel = mp.matrix([mp.mpf('-0.5') * x_2_dot, mp.mpf('-0.5') * y_2_dot, mp.mpf('0.0'),  # first
                          x_2_dot, y_2_dot,  mp.mpf('0.0'),  # second
                          mp.mpf('-0.5') * x_2_dot, mp.mpf('-0.5') * y_2_dot, mp.mpf('0.0')])  # third
    init_vel_p = init_vel * masses[0]
    x_10 = masses[0] * masses[0] * (mp.mpf('2.0') * np.sqrt(mp.mpf('2.0')) + 1) / \
           (np.sqrt(mp.mpf('2.0')) * mp.mpf('-0.5') - np.sum(np.dot(init_vel, init_vel)) / 2 / masses[0])
    x_10 = mp.mpf('5') * masses[0] * masses[0] / (1 + masses[0] * np.dot(init_vel, init_vel))
    x_10 = mp.mpf(str(x_10))
    init_pos = mp.matrix([-x_10, mp.mpf('0.0'), mp.mpf('0.0'), # first
                          mp.mpf('0.0'), mp.mpf('0.0'), mp.mpf('0.0'),  # second
                          x_10, mp.mpf('0.0'), mp.mpf('0.0')])  # third
    h = mp.mpf('1e-3')

    # n = 3000
    # n = 1650 примерный период
    n = 1680
    stormlet_verlet_explicit(init_pos, init_vel_p, masses, step=h, n=n, to_file=True, filename='result.dat')
    data_frame = pd.read_csv('result.dat', delimiter=' ', header=None)
    # plt.plot(data_frame[1], data_frame[2], label='first')
    # plt.legend()
    # plt.show()
    plt.plot(data_frame[0], (data_frame[4] - data_frame[1]) * (data_frame[8] -
                                                               0.5 * (data_frame[2] + data_frame[5])) -
             (data_frame[5] - data_frame[2]) * (data_frame[7] - 0.5 * (data_frame[1] + data_frame[4])),
             label='ksi_3 (t)')
    plt.legend()
    plt.show()
    ksi_3_values = 2 / np.sqrt(3) * (data_frame[4] - data_frame[1]) * (data_frame[8] -
                                                               0.5 * (data_frame[2] + data_frame[5])) - \
                   (data_frame[5] - data_frame[2]) * (data_frame[7] - 0.5 * (data_frame[1] + data_frame[4]))
    ksi_2_values = 2 / np.sqrt(3) * ((data_frame[4] - data_frame[1]) * (data_frame[7] - 0.5 * (data_frame[1] + data_frame[4])) +
                                     (data_frame[5] - data_frame[2]) * (data_frame[8] - 0.5 * (data_frame[2] + data_frame[5])))
    alpha_1 = data_frame[1]
    alpha_2 = data_frame[2]
    beta_1 = data_frame[4]
    beta_2 = data_frame[5]
    gamma_1 = data_frame[7]
    gamma_2 = data_frame[8]
    ksi_1_values = 0.5 * ((beta_1 - alpha_1) ** 2 + (beta_2 - alpha_2) ** 2) - 0.666666 * ((gamma_1 - 0.5 * (alpha_1 + beta_1)) ** 2 +
                                                                                           (gamma_2 - 0.5 * (alpha_2 + beta_2)) ** 2)
    plt.plot(data_frame[0], ksi_2_values, label='ksi_2(t)')
    plt.legend()
    plt.show()
    plt.plot(data_frame[0], ksi_1_values, label='ksi_1')
    plt.legend()
    plt.show()
    plt.plot(data_frame[0], ksi_1_values **2 + ksi_2_values**2 + ksi_3_values **2, label='sum of squares')
    plt.legend()
    plt.show()
    plt.plot(data_frame[0], ksi_1_values**2 + ksi_2_values**2, label='ksi_2 + ksi_1')
    plt.legend()
    plt.show()
    plt.plot(data_frame[0], ksi_3_values / ksi_2_values, label='ksi_3 / ksi_2')
    plt.legend()
    plt.show()
    plt.plot(data_frame[0], ksi_1_values, label='ksi_1')
    plt.plot(data_frame[0], ksi_2_values, label='ksi_2')
    plt.plot(data_frame[0], ksi_3_values, label='ksi_3')
    plt.plot([0, 1.7], [0, 0])
    plt.legend()
    plt.show()
    # t_min_index = np.argmin(ksi_3_values)
    # print(data_frame.loc(4, t_min_index))
    # plt.plot([data_frame[4, t_min_index], data_frame[4, t_min_index+1]], [data_frame[5, t_min_index],data_frame[5, t_min_index+ 1]], '0', label='second')
    # plt.plot([data_frame[1, t_min_index], data_frame[1, t_min_index+1]], [data_frame[2, t_min_index],data_frame[2, t_min_index+ 1]], '0', label='first')
    # plt.plot([data_frame[7, t_min_index], data_frame[7, t_min_index+1]], [data_frame[8, t_min_index],data_frame[8, t_min_index+ 1]], '0', label='second')
    # plt.legend()
    # plt.show()
    plt.plot(data_frame[1], data_frame[2], label='first')
    # plt.legend()
    # plt.show()
    plt.plot(data_frame[4], data_frame[5], label='second')
    # plt.legend()
    # plt.show()
    plt.plot(data_frame[7], data_frame[8], label='third')
    plt.legend()
    plt.show()

    index = np.argmin(abs(data_frame[1] - init_pos[0]) + abs(data_frame[2] - init_pos[1]) +
                      abs(data_frame[4] - init_pos[3]) + abs(data_frame[5] - init_pos[4]) +
                      abs(data_frame[7] - init_pos[6]) + abs(data_frame[8] - init_pos[7]))
    value = np.min(abs(data_frame[1] - init_pos[0]) + abs(data_frame[2] - init_pos[1]) +
                      abs(data_frame[4] - init_pos[3]) + abs(data_frame[5] - init_pos[4]) +
                      abs(data_frame[7] - init_pos[6]) + abs(data_frame[8] - init_pos[7]))
    print(index, '       ', value)

    # h = mp.mpf(args.step)
    # stormlet_verlet_explicit(init_pos, init_vel, masses, step=h, n=n, to_file=True, filename='result.dat')
    # data_frame = pd.read_csv('result.dat', delimiter=' ', header=None)
    # #plt.plot(data_frame[1], data_frame[2])
    # #plt.plot(data_frame[4], data_frame[5])
    # plt.plot(data_frame[7], data_frame[8])
    # plt.show()
    # a_e_list = [2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
    # for distance in a_e_list:
    #     init_pos = mp.matrix([mp.mpf('0.0'), mp.mpf('0.0'), mp.mpf('0.0'),  # Солнце
    #                           mp.mpf(str(-sun_jup)), mp.mpf('0.0'), mp.mpf('0.0'),  # Юпитер
    #                           mp.mpf(str(-distance * a_e)), mp.mpf('0.0'), mp.mpf('0.0')])  # астероид
    #     first_cosmic = np.sqrt(G * masses[0] / (a_e * distance))
    #     init_vel = mp.matrix(([mp.mpf('0.0'), mp.mpf('0.0'), mp.mpf('0.0'),  # Солнце скорость в км/д
    #                            mp.mpf('0.0'), mp.mpf(str(jup_vel)) * masses[1], mp.mpf('0.0'),  # Юпитер
    #                            mp.mpf('0.0'), mp.mpf(str(first_cosmic)) * masses[2], mp.mpf('0.0')]))
    #     stormlet_verlet_explicit(init_pos, init_vel, masses, step=h, n=n, to_file=True,
    #                              filename=f'result_{distance}.dat')
    #     data_frame = pd.read_csv(f'result_{distance}.dat', delimiter=' ', header=None)
    #     plt.plot(data_frame[7], data_frame[8], label=f'{distance}_a_e')
    #     plt.legend()
    #     plt.show()