#include <cstring>

#define MAX_LEVELS 1000

template <class dataType>
struct MTX{
    int rows;
    int cols;
    int nnz;
    int *row;
    int *col;
    dataType* data;
};

//Func: 按照行从小到大，行相等时，列从小到大的顺序
template<class dataType>
bool if_sorted_coo(MTX<dataType>* mtx)
{
    int nnz = mtx->nnz;
    for (int i = 0; i < nnz - 1; i++)
    {
        if ((mtx->row[i] > mtx->row[i+1]) || (mtx->row[i] == mtx->row[i+1] 
              && mtx->col[i] > mtx->col[i+1]))
            return false;
    }
    return true;
}

template<class dataType>
bool sort_coo(MTX<dataType>* mtx)
{

    int i = 0;
    int beg[MAX_LEVELS], end[MAX_LEVELS], L, R ;
    int pivrow, pivcol;
    dataType pivdata;

    beg[0]=0; 
    end[0]=mtx->nnz;
    while (i>=0) 
    {
        L=beg[i];
        if (end[i] - 1 > end[i])
            R = end[i];
        else
            R = end[i] - 1;
        if (L<R) 
        {
            int middle = (L+R)/2;
            pivrow=mtx->row[middle]; 
            pivcol=mtx->col[middle];
            pivdata=mtx->data[middle];
            mtx->row[middle] = mtx->row[L];
            mtx->col[middle] = mtx->col[L];
            mtx->data[middle] = mtx->data[L];
            mtx->row[L] = pivrow;
            mtx->col[L] = pivcol;
            mtx->data[L] = pivdata;
            if (i==MAX_LEVELS-1) 
                return false;
            while (L<R) 
            {
                while (((mtx->row[R] > pivrow) || 
                            (mtx->row[R] == pivrow && mtx->col[R] > pivcol)) 
                        && L<R) 
                    R--; 
                if (L<R) 
                {
                    mtx->row[L] = mtx->row[R];
                    mtx->col[L] = mtx->col[R];
                    mtx->data[L] = mtx->data[R];
                    L++;
                }
                while (((mtx->row[L] < pivrow) || 
                            (mtx->row[L] == pivrow && mtx->col[L] < pivcol)) 
                        && L<R) 
                    L++; 
                if (L<R) 
                {
                    mtx->row[R] = mtx->row[L];
                    mtx->col[R] = mtx->col[L];
                    mtx->data[R] = mtx->data[L];
                    R--;
                }
            }
            mtx->row[L] = pivrow;
            mtx->col[L] = pivcol;
            mtx->data[L] = pivdata;
            beg[i+1]=L+1; 
            end[i+1]=end[i]; 
            end[i++]=L; 
        }
        else 
        {
            i--; 
        }
    }
    return true;
}

//FUNC: 按照column从小到大，col相等时，row从小到大的顺序
template<class dataType>
bool if_sorted_col_coo(MTX<dataType>* mtx)
{
    int nnz = mtx->nnz;
    for (int i = 0; i < nnz - 1; i++)
    {
        if ((mtx->col[i] > mtx->col[i+1]) || (mtx->col[i] == mtx->col[i+1] 
              && mtx->row[i] > mtx->row[i+1]))
            return false;
    }
    return true;
}

template<class dataType>
bool sort_col_coo(MTX<dataType>* mtx)
{

    int i = 0;
    int beg[MAX_LEVELS], end[MAX_LEVELS], L, R ;
    int pivrow, pivcol;
    dataType pivdata;

    beg[0]=0; 
    end[0]=mtx->nnz;
    while (i>=0) 
    {
        L=beg[i];
        if (end[i] - 1 > end[i])
            R = end[i];
        else
            R = end[i] - 1;
        if (L<R) 
        {
            int middle = (L+R)/2;
            pivrow=mtx->row[middle]; 
            pivcol=mtx->col[middle];
            pivdata=mtx->data[middle];
            mtx->row[middle] = mtx->row[L];
            mtx->col[middle] = mtx->col[L];
            mtx->data[middle] = mtx->data[L];
            mtx->row[L] = pivrow;
            mtx->col[L] = pivcol;
            mtx->data[L] = pivdata;
            if (i==MAX_LEVELS-1) 
                return false;
            while (L<R) 
            {
                while (((mtx->col[R] > pivcol) || 
                            (mtx->col[R] == pivcol && mtx->row[R] > pivrow)) 
                        && L<R) 
                    R--; 
                if (L<R) 
                {
                    mtx->row[L] = mtx->row[R];
                    mtx->col[L] = mtx->col[R];
                    mtx->data[L] = mtx->data[R];
                    L++;
                }
                while (((mtx->col[L] < pivcol) || 
                            (mtx->col[L] == pivcol && mtx->row[L] < pivrow)) 
                        && L<R) 
                    L++; 
                if (L<R) 
                {
                    mtx->row[R] = mtx->row[L];
                    mtx->col[R] = mtx->col[L];
                    mtx->data[R] = mtx->data[L];
                    R--;
                }
            }
            mtx->row[L] = pivrow;
            mtx->col[L] = pivcol;
            mtx->data[L] = pivdata;
            beg[i+1]=L+1; 
            end[i+1]=end[i]; 
            end[i++]=L; 
        }
        else 
        {
            i--; 
        }
    }
    return true;
}

template<class dataType>
void fileToMtxCoo(const char* filename,MTX<dataType> *mtx, bool isRowSorted)
{
    FILE* infile = fopen(filename, "r");
    //Added by limin.
    if (infile == NULL) {
      printf("open file error\n");
      exit(1);
    }
    char tmpstr[100];
    char tmpline[1030];
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    bool ifreal = false;
    if (strcmp(tmpstr, "real") == 0)
        ifreal = true;
    bool ifsym = false;
    fscanf(infile, "%s", tmpstr);
    if (strcmp(tmpstr, "symmetric") == 0)
        ifsym = true;
    int height = 0;
    int width = 0;
    int nnz = 0;
    while (true)
    {
        fscanf(infile, "%s", tmpstr);
        if (tmpstr[0] != '%')
        {
            height = atoi(tmpstr);
            break;
        }
        fgets(tmpline, 1025, infile);
    }

    fscanf(infile, "%d %d", &width, &nnz);
    mtx->rows = height;
    mtx->cols = width;

    int* rows = (int*)malloc(sizeof(int)*nnz);
    int* cols = (int*)malloc(sizeof(int)*nnz);
    dataType* data = (dataType*)malloc(sizeof(dataType)*nnz);
    
    int diaCount = 0;
    for (int i = 0; i < nnz; i++)
    {
        int rowid = 0;
        int colid = 0;
        fscanf(infile, "%d %d", &rowid, &colid);
        rows[i] = rowid - 1;
        cols[i] = colid - 1;
        data[i] = i;
        if (ifreal)
        {
            double dbldata = 0.0f;
            fscanf(infile, "%lf", &dbldata);
            data[i] = (dataType)dbldata;
        }
        else
        {
            data[i] = 1.0;
        }
        if (rows[i] == cols[i])
            diaCount++;
    }
    
    if (ifsym)
    {
        int newnnz = nnz * 2 - diaCount;
        mtx->nnz = newnnz;
        mtx->row = (int*)malloc(sizeof(int)*newnnz);
        mtx->col = (int*)malloc(sizeof(int)*newnnz);
        mtx->data = (dataType*)malloc(sizeof(dataType)*newnnz);
        int matid = 0;
        for (int i = 0; i < nnz; i++)
        {
            mtx->row[matid] = rows[i];
            mtx->col[matid] = cols[i];
            mtx->data[matid] = data[i];
            matid++;
            if (rows[i] != cols[i])
            {
                mtx->row[matid] = cols[i];
                mtx->col[matid] = rows[i];
                mtx->data[matid] = data[i];
                matid++;
            }
        }
        if(matid != newnnz){
            std::cout<<"Error: matid != newnnz!"<<std::endl;
        }
        bool tmp = false;
        if (isRowSorted) {
          sort_coo<dataType>(mtx);
          tmp=if_sorted_coo<dataType>(mtx);
        }
        else {
          sort_col_coo<dataType>(mtx);
          tmp=if_sorted_col_coo<dataType>(mtx);
        }
        if(tmp != true){
            std::cout<<"Error: not sorted!"<<std::endl;
        }
    }
    else
    {
        mtx->nnz = nnz;
        mtx->row = (int*)malloc(sizeof(int)*nnz);
        mtx->col = (int*)malloc(sizeof(int)*nnz);
        mtx->data = (dataType*)malloc(sizeof(dataType)*nnz);
        memcpy(mtx->row, rows, sizeof(int)*nnz);
        memcpy(mtx->col, cols, sizeof(int)*nnz);
        memcpy(mtx->data, data, sizeof(dataType)*nnz);
        bool tmp = false;
        if (isRowSorted) {
          if (!if_sorted_coo(mtx))
            sort_coo<dataType>(mtx);
          tmp=if_sorted_coo<dataType>(mtx);
        } else {
          if (!if_sorted_col_coo(mtx))
            sort_col_coo<dataType>(mtx);
          tmp=if_sorted_col_coo<dataType>(mtx);
        }

        if(tmp != true){
            std::cout<<"Error: not sorted!"<<std::endl;
        }
    }
    fclose(infile);
    free(rows);
    free(cols);
    free(data);
    return;    
}

template<class dataType>
void printMtx(MTX<dataType> *mtx)
{
    std::cout<<"rows:"<<mtx->rows<<"  cols:"<<mtx->cols<<"  non zeros:"<<mtx->nnz<<std::endl;
    for(int i=0;i<mtx->nnz;i++){
       std::cout<<mtx->col[i]<<"  "<<mtx->row[i]<<"  "<<mtx->data[i]<<std::endl;
    }
}

