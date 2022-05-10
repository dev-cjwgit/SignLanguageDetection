import sqlite3


class DatasetWriter:
    create_action_table_sql = 'CREATE TABLE if not exists action(uid INTEGER PRIMARY KEY AUTOINCREMENT, ' \
                              'action_num INTEGER, ' \
                              'movie_num INTEGER, ' \
                              'UNIQUE(action_num, movie_num))'

    create_frame_data_table_sql = 'CREATE TABLE if not exists frame(uid INTEGER PRIMARY KEY AUTOINCREMENT, ' \
                                  'action_uid INTEGER, ' \
                                  'frame_num INTEGER, ' + \
                                  ', '.join(['p%d REAL' % (idx) for idx in range(258)]) + ', ' + \
                                  'UNIQUE(action_uid, frame_num)' \
                                  'CONSTRAINT action_uid_fk FOREIGN KEY(action_uid) REFERENCES action(uid) ' \
                                  'ON DELETE CASCADE);'

    def __init__(self, file_name):
        self.conn = sqlite3.connect(file_name)
        self.cur = self.conn.cursor()
        self.cur.execute(self.create_action_table_sql)
        self.cur.execute(self.create_frame_data_table_sql)
        self.cur.execute('PRAGMA foreign_keys = ON')
        self.conn.commit()

    def append(self, action: int, data):
        self.cur.execute('SELECT min(movie_num) FROM action WHERE action_num = ' + str(action))
        min_movie_num = self.cur.fetchone()[0]
        if min_movie_num != 0:
            next_movie_num = 0
        else:
            self.cur.execute(
                'SELECT min(movie_num) + 1 '
                'FROM action '
                'WHERE action_num = ' + str(action) + ' AND (movie_num + 1) '
                                                      'NOT IN (SELECT movie_num '
                                                      'FROM action '
                                                      'WHERE action_num = ' + str(action) + ');')

            next_movie_num = self.cur.fetchone()[0]
            if next_movie_num is None:
                next_movie_num = 0

        self.cur.execute('INSERT INTO action (action_num, movie_num) '
                         'VALUES(' + str(action) + ', ' + str(next_movie_num) + ')')

        self.cur.execute(
            'SELECT uid FROM action WHERE action_num = ' + str(action) + ' AND movie_num = ' + str(
                next_movie_num) + ';')

        action_uid = self.cur.fetchone()[0]

        for frame_num, frame_data in enumerate(data):
            temp_sql = 'INSERT INTO frame (action_uid, frame_num, ' + \
                       ', '.join(['p%d' % (idx) for idx in range(258)]) + ')' + \
                       ' VALUES (' + str(action_uid) + ', ' + str(frame_num) + ', ' + \
                       ', '.join([str(val) for val in frame_data]) + ')'
            self.cur.execute(temp_sql)
        self.conn.commit()

    def delete(self, action, num):
        self.cur.execute('DELETE FROM action WHERE action_num = ' + str(action) + ' AND movie_num = ' + str(num) + ';')
        self.conn.commit()

    def __del__(self):
        self.cur.close()
        self.conn.close()
