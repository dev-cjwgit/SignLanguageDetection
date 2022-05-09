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
                                  'CONSTRAINT action_uid_fk FOREIGN KEY(action_uid) REFERENCES action(uid));'

    def __init__(self, file_name):
        self.conn = sqlite3.connect(file_name)
        self.cur = self.conn.cursor()
        self.cur.execute(self.create_action_table_sql)
        self.cur.execute(self.create_frame_data_table_sql)
        self.conn.commit()

    def append(self, action: int, count: int, data):
        self.cur.execute('INSERT INTO action (action_num, movie_num) '
                         'VALUES(' + str(action) + ', ' + str(count) + ')')

        self.cur.execute(
            'SELECT uid FROM action WHERE action_num = ' + str(action) + ' AND movie_num = ' + str(count) + ';')

        action_uid = self.cur.fetchone()[0]

        for frame_num, frame_data in enumerate(data):
            temp_sql = 'INSERT INTO frame (action_uid, frame_num, ' + \
                       ', '.join(['p%d' % (idx) for idx in range(258)]) + ')' + \
                       ' VALUES (' + str(action_uid) + ', ' + str(frame_num) + ', ' + \
                       ', '.join([str(val) for val in frame_data]) + ')'
            self.cur.execute(temp_sql)
        self.conn.commit()

    def __del__(self):
        self.cur.close()
        self.conn.close()
